//! The kredo daemon: HTTP API and model scheduler.
//!
//! The scheduler owns a map of loaded engines with keep-alive expiry and a
//! per-model concurrency pool. The TypeSafe endpoints (`/v1/systemone`,
//! `/v1/decisions`, `GET /v1/models`) are wire-compatible; `/api/*` is
//! jeff-native. Production knobs live in [`Config`].

pub mod config;
pub mod limit;
pub mod metrics;
pub mod ui;

pub use config::{Config, LogFormat};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kredo_api::{
    ApiError, DecideRequest, DecideResponse, DecisionsRequest, ModelInfo, PsResponse, PullProgress,
    PullRequest, Question, Routing, RunningModel, ShowRequest, ShowResponse, SystemOneRequest,
    SystemOneResponse, TagsResponse,
};
use kredo_lang as lang;
use kredo_registry::{Manifest, Registry};
use kredo_runner::{Device, Engine};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

pub const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(5 * 60);
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

static REQUEST_COUNTER: LazyLock<std::sync::atomic::AtomicU64> =
    LazyLock::new(|| std::sync::atomic::AtomicU64::new(0));

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct Loaded {
    engine: Arc<Mutex<Box<dyn Engine>>>,
    name: String,
    full_name: String,
    questions: Vec<Question>,
    size: u64,
    expires_at: Instant,
    /// Per-model inference pool.
    permits: Arc<Semaphore>,
}

pub struct ServerState {
    registry: Registry,
    cfg: Config,
    loaded: Mutex<HashMap<String, Loaded>>,
    limiter: limit::RateLimiter,
    metrics: Arc<metrics::Metrics>,
}

impl ServerState {
    pub fn new(cfg: Config) -> Arc<Self> {
        let registry = match &cfg.models_dir {
            Some(dir) => Registry::with_root(dir.clone()),
            None => Registry::open(),
        };
        Self::with_registry_and_metrics(cfg, registry)
    }

    /// Explicit registry (used by tests and embedders).
    pub fn new_with_registry(cfg: Config, models_dir: std::path::PathBuf) -> Arc<Self> {
        Self::with_registry_and_metrics(cfg, Registry::with_root(models_dir))
    }

    fn with_registry_and_metrics(cfg: Config, registry: Registry) -> Arc<Self> {
        let limiter = limit::RateLimiter::new(cfg.rate_limit_per_min);
        Arc::new(Self {
            registry,
            limiter,
            cfg,
            loaded: Mutex::new(HashMap::new()),
            metrics: metrics::Metrics::new(),
        })
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn metrics(&self) -> &Arc<metrics::Metrics> {
        &self.metrics
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SrvError {
    #[error("{0}")]
    Api(ApiError),
    #[error(transparent)]
    Registry(#[from] kredo_registry::RegistryError),
    #[error(transparent)]
    Runner(#[from] kredo_runner::RunnerError),
    #[error("verification: {0}")]
    Verify(#[from] kredo_runner::verification::VerifyError),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<ApiError> for SrvError {
    fn from(e: ApiError) -> Self {
        SrvError::Api(e)
    }
}

impl IntoResponse for SrvError {
    fn into_response(self) -> Response {
        let msg = self.to_string();
        let status = match self {
            SrvError::Registry(kredo_registry::RegistryError::NotFound(_)) => StatusCode::NOT_FOUND,
            SrvError::Api(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(ApiError::new(msg))).into_response()
    }
}

pub type SrvResult<T> = Result<T, SrvError>;

fn busy_error(model: &str) -> SrvError {
    SrvError::Api(ApiError::new(format!(
        "model {model} is at its concurrency limit; retry shortly"
    )))
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

impl ServerState {
    async fn get_loaded(&self, full_name: &str) -> SrvResult<()> {
        {
            let mut loaded = self.loaded.lock().await;
            if let Some(entry) = loaded.get_mut(full_name) {
                entry.expires_at = Instant::now() + self.cfg.keep_alive;
                return Ok(());
            }
        }

        let mut loaded = self.loaded.lock().await;
        if let Some(entry) = loaded.get_mut(full_name) {
            entry.expires_at = Instant::now() + self.cfg.keep_alive;
            return Ok(());
        }
        let local = self.registry.get(full_name)?;
        if self.cfg.require_verified {
            let report = kredo_runner::verification::verify(&local, &self.registry)?;
            if !report.passed {
                return Err(SrvError::Api(ApiError::new(format!(
                    "model {full_name} failed verification; refusing to load"
                ))));
            }
            tracing::info!(model = full_name, "verified on load");
        }
        let engine = kredo_runner::load(&local, &self.registry, Device::from_env())?;
        let permits = Arc::new(Semaphore::new(self.cfg.max_concurrency));
        loaded.insert(
            full_name.to_string(),
            Loaded {
                engine: Arc::new(Mutex::new(engine)),
                name: local.manifest.name.clone(),
                full_name: local.manifest.full_name(),
                questions: local.manifest.questions.clone(),
                size: local.size,
                expires_at: Instant::now() + self.cfg.keep_alive,
                permits,
            },
        );
        self.metrics
            .models_loaded
            .store(loaded.len() as u64, Ordering::Relaxed);
        tracing::info!(model = full_name, "loaded");
        Ok(())
    }

    /// The manifest of `spec`, from the embedded library or a pulled model.
    fn manifest_of(&self, spec: &str) -> SrvResult<Manifest> {
        let (name, tag) = kredo_registry::split_name_tag(spec)?;
        if let Some(m) = kredo_registry::library::lookup(&name, &tag) {
            return Ok(m);
        }
        Ok(self.registry.get(spec)?.manifest)
    }
}

// ---------------------------------------------------------------------------
// Core decision logic
// ---------------------------------------------------------------------------

async fn decide_core(
    state: &ServerState,
    model_spec: Option<&str>,
    state_value: &serde_json::Value,
    questions: Option<Vec<Question>>,
) -> SrvResult<(String, Vec<kredo_api::Answer>, Routing)> {
    // Input hardening: bound the state text.
    let text = kredo_api::state_to_text(state_value);
    if text.len() > state.cfg.body_limit {
        return Err(ApiError::new("state too large").into());
    }
    let spec = model_spec.unwrap_or("kredo");

    let route_start = Instant::now();
    let manifest = state.manifest_of(spec)?;
    let resolved = match &manifest.router {
        Some(router) => lang::route(&text, &router.en, &router.multilingual).tag,
        None => manifest.full_name(),
    };
    let script = lang::detect_script(&text);
    let route_ms = route_start.elapsed().as_secs_f64() * 1000.0;

    state.get_loaded(&resolved).await?;

    // Snapshot engine handle, permit pool and default questions without
    // holding the map lock across inference.
    let (engine, permits, name, questions) = {
        let loaded = state.loaded.lock().await;
        let entry = loaded
            .get(&resolved)
            .ok_or_else(|| SrvError::Api(ApiError::new(format!("model {resolved} vanished"))))?;
        let questions = questions.unwrap_or_else(|| entry.questions.clone());
        (
            entry.engine.clone(),
            entry.permits.clone(),
            entry.name.clone(),
            questions,
        )
    };

    let infer_start = Instant::now();
    let _permit = tokio::time::timeout(Duration::from_secs(30), permits.acquire_owned())
        .await
        .map_err(|_| busy_error(&resolved))?
        .map_err(|_| busy_error(&resolved))?;
    let answers = {
        let eng = engine.lock().await;
        eng.decide(&text, &questions)?
    };
    let infer_ms = infer_start.elapsed().as_secs_f64();
    state
        .metrics
        .inference_duration
        .observe(Duration::from_secs_f64(infer_ms));
    state
        .metrics
        .decisions_total
        .fetch_add(1, Ordering::Relaxed);

    let language = lang::route(&text, "en", "multi").language;
    let routing = Routing {
        resolved,
        script: Some(format!("{script:?}").to_lowercase()),
        language: Some(language),
        route_ms,
    };
    Ok((name, answers, routing))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn check_auth(state: &ServerState, headers: &HeaderMap) -> SrvResult<()> {
    use subtle::ConstantTimeEq;
    if let Some(expected) = &state.cfg.api_key {
        let got = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or_default();
        let ok = expected.as_bytes().ct_eq(got.as_bytes());
        if !bool::from(ok) {
            return Err(SrvError::Api(ApiError::new("invalid or missing API key")));
        }
    }
    Ok(())
}

fn client_key(state: &ServerState, headers: &HeaderMap) -> String {
    if state.cfg.api_key.is_some() {
        if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
            return v.to_string();
        }
    }
    headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("local")
        .to_string()
}

async fn healthz() -> &'static str {
    "ok"
}

async fn readyz(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let models = state.registry.list().map(|m| m.len()).unwrap_or(0);
    Json(serde_json::json!({"ready": true, "models": models}))
}

async fn version() -> impl IntoResponse {
    Json(serde_json::json!({"version": VERSION}))
}

async fn metrics_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    (
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
}

async fn v1_systemone(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<SystemOneRequest>,
) -> SrvResult<Json<SystemOneResponse>> {
    check_auth(&state, &headers).await?;
    let start = Instant::now();
    let (model, answers, _routing) = decide_core(
        &state,
        req.model.as_deref(),
        &req.state,
        Some(req.questions).filter(|q| !q.is_empty()),
    )
    .await?;
    Ok(Json(SystemOneResponse {
        model,
        answers,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn v1_decisions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<DecisionsRequest>,
) -> SrvResult<Json<Vec<SystemOneResponse>>> {
    check_auth(&state, &headers).await?;
    if req.states.len() > 64 {
        return Err(ApiError::new("at most 64 states per batch").into());
    }
    let mut out = Vec::with_capacity(req.states.len());
    for s in &req.states {
        let start = Instant::now();
        let (model, answers, _) =
            decide_core(&state, req.model.as_deref(), s, Some(req.questions.clone())).await?;
        out.push(SystemOneResponse {
            model,
            answers,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        });
    }
    Ok(Json(out))
}

fn model_info(
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    questions: Vec<Question>,
) -> ModelInfo {
    ModelInfo {
        name,
        model,
        modified_at,
        size,
        questions,
    }
}

async fn v1_models(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> SrvResult<Json<Vec<ModelInfo>>> {
    check_auth(&state, &headers).await?;
    let models = state
        .registry()
        .list()?
        .into_iter()
        .map(|m| {
            let full = m.manifest.full_name();
            model_info(
                m.manifest.name.clone(),
                full,
                m.modified_at,
                m.size,
                m.manifest.questions,
            )
        })
        .collect();
    Ok(Json(models))
}

async fn api_decide(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<DecideRequest>,
) -> SrvResult<Json<DecideResponse>> {
    check_auth(&state, &headers).await?;
    let start = Instant::now();
    let (model, answers, routing) = decide_core(
        &state,
        Some(&req.model),
        &req.state,
        Some(req.questions).filter(|q| !q.is_empty()),
    )
    .await?;
    Ok(Json(DecideResponse {
        model,
        answers,
        routing,
        load_ms: 0.0,
        infer_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn api_tags(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> SrvResult<Json<TagsResponse>> {
    check_auth(&state, &headers).await?;
    let models = state
        .registry()
        .list()?
        .into_iter()
        .map(|m| {
            let full = m.manifest.full_name();
            model_info(
                m.manifest.name.clone(),
                full,
                m.modified_at,
                m.size,
                m.manifest.questions,
            )
        })
        .collect();
    Ok(Json(TagsResponse { models }))
}

async fn api_show(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<ShowRequest>,
) -> SrvResult<Json<ShowResponse>> {
    check_auth(&state, &headers).await?;
    let m = state.registry().get(&req.model)?;
    let mut details = std::collections::BTreeMap::new();
    details.insert("engine".into(), m.manifest.engine.clone().into());
    details.insert("precision".into(), m.manifest.precision.clone().into());
    details.insert(
        "max_seq_len".into(),
        serde_json::Value::Number(m.manifest.max_seq_len.into()),
    );
    details.insert("description".into(), m.manifest.description.clone().into());
    let verified = m
        .manifest
        .verification
        .as_ref()
        .map(|v| v.fixture_digest.len() == 64);
    details.insert(
        "verified".into(),
        serde_json::Value::Bool(verified.unwrap_or(false)),
    );
    let full = m.manifest.full_name();
    let provenance = m
        .manifest
        .provenance
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| SrvError::Api(ApiError::new(e.to_string())))?;
    let verification = m
        .manifest
        .verification
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| SrvError::Api(ApiError::new(e.to_string())))?;
    Ok(Json(ShowResponse {
        name: m.manifest.name,
        model: full,
        modified_at: m.modified_at,
        size: m.size,
        details,
        questions: m.manifest.questions,
        provenance,
        verification,
    }))
}

async fn api_ps(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> SrvResult<Json<PsResponse>> {
    check_auth(&state, &headers).await?;
    let loaded = state.loaded.lock().await;
    let models = loaded
        .values()
        .map(|l| {
            let remaining = l
                .expires_at
                .saturating_duration_since(Instant::now())
                .as_secs();
            RunningModel {
                name: l.name.clone(),
                model: l.full_name.clone(),
                size: l.size,
                expires_at: format!("in {remaining}s"),
            }
        })
        .collect();
    Ok(Json(PsResponse { models }))
}

async fn api_delete(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<ShowRequest>,
) -> SrvResult<StatusCode> {
    check_auth(&state, &headers).await?;
    state.loaded.lock().await.remove(&req.model);
    state.registry().remove(&req.model)?;
    Ok(StatusCode::OK)
}

async fn api_stop(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<ShowRequest>,
) -> SrvResult<StatusCode> {
    check_auth(&state, &headers).await?;
    state.loaded.lock().await.remove(&req.model);
    Ok(StatusCode::OK)
}

async fn api_pull(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<PullRequest>,
) -> SrvResult<Response> {
    check_auth(&state, &headers).await?;
    let manifest: Manifest = kredo_registry::Manifest::from_library(&req.model)
        .or_else(|_| state.registry().get(&req.model).map(|l| l.manifest))?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<PullProgress>();
    let registry = Registry::with_root(state.registry().root().to_path_buf());
    tokio::spawn(async move {
        let mut on_progress = |p: PullProgress| {
            let _ = tx.send(p);
        };
        let result = kredo_registry::pull::pull(&manifest, &registry, &mut on_progress).await;
        let _ = tx.send(PullProgress {
            status: match result {
                Ok(_) => "success".into(),
                Err(e) => format!("error: {e}"),
            },
            digest: None,
            total: None,
            completed: None,
        });
        drop(tx);
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|p| {
            let mut line = serde_json::to_string(&p).unwrap_or_default();
            line.push('\n');
            (Ok::<_, std::convert::Infallible>(line), rx)
        })
    });
    let body = Body::from_stream(stream);
    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(body)
        .unwrap())
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// Request ID + metrics + rate limiting in one pass.
async fn observability(
    State(state): State<Arc<ServerState>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let id = format!(
        "req-{:x}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed),
    );

    if !state.limiter.allow(&client_key(&state, req.headers())) {
        state
            .metrics
            .rate_limited_total
            .fetch_add(1, Ordering::Relaxed);
        state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("retry-after", "5")],
            Json(ApiError::new("rate limit exceeded")),
        )
            .into_response();
    }

    req.headers_mut()
        .insert("x-request-id", id.parse().unwrap());

    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let resp = next.run(req).await;

    let elapsed = start.elapsed();
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    state.metrics.request_duration.observe(elapsed);
    if resp.status().is_server_error() {
        state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
    }
    tracing::info!(
        request_id = %id,
        method = %method,
        path = %path,
        status = resp.status().as_u16(),
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        "request"
    );

    let mut resp = resp;
    resp.headers_mut()
        .insert("x-request-id", id.parse().unwrap());
    resp
}

// ---------------------------------------------------------------------------
// Router + serve
// ---------------------------------------------------------------------------

pub fn router(state: Arc<ServerState>) -> Router {
    let timeout =
        TimeoutLayer::with_status_code(StatusCode::SERVICE_UNAVAILABLE, state.cfg.request_timeout);
    let body_limit = RequestBodyLimitLayer::new(state.cfg.body_limit);

    let app = Router::new()
        .route("/", get(healthz))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/version", get(version))
        .route("/metrics", get(metrics_handler))
        .route("/v1/systemone", post(v1_systemone))
        .route("/v1/decisions", post(v1_decisions))
        .route("/v1/models", get(v1_models))
        .route("/api/decide", post(api_decide))
        .route("/api/pull", post(api_pull))
        .route("/api/tags", get(api_tags))
        .route("/api/show", post(api_show))
        .route("/api/ps", get(api_ps))
        .route("/api/delete", post(api_delete))
        .route("/api/stop", post(api_stop))
        .layer(middleware::from_fn_with_state(state.clone(), observability))
        .layer(body_limit)
        .layer(timeout)
        .with_state(state.clone());

    // UI routes are stateless; merge after the stateful app is resolved.
    let app = Router::new().merge(ui::routes()).merge(app);

    if state.cfg.cors {
        app.layer(tower_http::cors::CorsLayer::permissive())
    } else {
        app
    }
}

/// Periodic housekeeping: keep-alive expiry + rate-limiter sweep.
pub async fn reap_loop(state: Arc<ServerState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    loop {
        interval.tick().await;
        let swept = state.limiter.sweep();
        if swept > 0 {
            tracing::debug!(swept, "rate limiter buckets swept");
        }
        let mut loaded = state.loaded.lock().await;
        let now = Instant::now();
        loaded.retain(|k, v| {
            if v.expires_at > now {
                true
            } else {
                tracing::info!(model = %k, "unloaded (keep-alive expired)");
                false
            }
        });
        state
            .metrics
            .models_loaded
            .store(loaded.len() as u64, Ordering::Relaxed);
    }
}

/// Run the daemon until shutdown is signalled. Handles TLS when configured.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    cfg.validate().map_err(anyhow::Error::msg)?;
    let state = ServerState::new(cfg.clone());
    let app = router(state.clone());

    if cfg.is_network_exposed() {
        tracing::warn!(
            addr = %cfg.host,
            "binding a non-loopback address; ensure TLS and KREDO_API_KEY are configured"
        );
    }

    let shutdown = shutdown_signal();
    match (&cfg.tls_cert, &cfg.tls_key) {
        (Some(cert), Some(key)) => {
            let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
            tracing::info!(addr = %cfg.host, version = VERSION, "kredo daemon (TLS) listening");
            let handle = axum_server::Handle::new();
            let h2 = handle.clone();
            tokio::spawn(async move {
                shutdown.await;
                h2.graceful_shutdown(Some(Duration::from_secs(30)));
            });
            axum_server::bind_rustls(cfg.host, tls)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        _ => {
            tracing::info!(addr = %cfg.host, version = VERSION, "kredo daemon listening");
            let listener = tokio::net::TcpListener::bind(cfg.host).await?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await?;
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received; draining");
}
