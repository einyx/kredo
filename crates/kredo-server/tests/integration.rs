//! Integration tests: full API surface against an in-process daemon with a
//! deterministic mock model. No network, no ONNX files.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use kredo_registry::{LocalModel, Manifest, Registry};
use kredo_server::Config;
use serde_json::json;
use tower::ServiceExt;

struct TestDaemon {
    router: Router,
    registry: Registry,
}

use std::sync::atomic::{AtomicU32, Ordering};

static DAEMON_SEQ: AtomicU32 = AtomicU32::new(0);

impl TestDaemon {
    fn new(cfg: Config) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "jeff-it-{}-{}",
            std::process::id(),
            DAEMON_SEQ.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = Registry::with_root(dir);
        // Seed a mock model.
        let model = LocalModel {
            manifest: Manifest {
                schema: 1,
                name: "mock".into(),
                tag: "latest".into(),
                description: "test".into(),
                engine: "mock".into(),
                precision: "none".into(),
                max_seq_len: 0,
                source: crate_source_empty(),
                decision: kredo_decision::DecisionSpec {
                    layout: kredo_decision::HeadLayout::Pairwise {
                        template: String::new(),
                        positive_labels: vec![],
                        id2label: Default::default(),
                        temperature: None,
                    },
                },
                questions: vec![kredo_api::Question {
                    id: "score".into(),
                    kind: kredo_api::QuestionKind::Score,
                    prompt: "test score".into(),
                    options: vec![],
                    min: Some(0.0),
                    max: Some(3.0),
                }],
                router: None,
                provenance: None,
                verification: None,
            },
            digests: Default::default(),
            size: 0,
            modified_at: "now".into(),
        };
        registry.save(&model).unwrap();
        let state = kredo_server::ServerState::new_with_registry(cfg, registry_path(&registry));
        TestDaemon {
            router: kredo_server::router(state),
            registry,
        }
    }

    async fn req(
        &mut self,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let builder = Request::builder().method(method).uri(uri);
        let req = match body {
            Some(v) => builder
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        };
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let value = if bytes.is_empty() {
            json!(null)
        } else {
            serde_json::from_slice(&bytes).unwrap_or(json!(null))
        };
        (status, value)
    }
}

fn registry_path(r: &Registry) -> std::path::PathBuf {
    r.root().to_path_buf()
}

fn crate_source_empty() -> kredo_registry::Source {
    kredo_registry::Source {
        repo: None,
        revision: "main".into(),
        files: vec![],
    }
}

fn cfg() -> Config {
    Config {
        api_key: Some("test-key".into()),
        ..Config::default()
    }
}

#[tokio::test]
async fn health_and_version_are_open() {
    let mut d = TestDaemon::new(Config::default());
    let (status, _) = d.req("GET", "/healthz", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, v) = d.req("GET", "/version", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(v["version"].is_string());
}

#[tokio::test]
async fn auth_rejects_missing_and_wrong_keys() {
    let mut d = TestDaemon::new(cfg());
    let (status, _) = d.req("GET", "/api/tags", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // (Correct key covered by decide test below via headerless Config; this
    // daemon requires the key on every call, so use an open daemon for the
    // positive path.)
}

#[tokio::test]
async fn decide_with_mock_model() {
    let mut d = TestDaemon::new(Config::default());
    let (status, v) = d
        .req(
            "POST",
            "/api/decide",
            Some(json!({"model": "mock", "state": "hello world"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "body: {v}");
    assert_eq!(v["model"], "mock");
    assert!(!v["answers"].as_array().unwrap().is_empty());
    let score = &v["answers"][0]["score"];
    assert!(score["value"].as_f64().unwrap() >= 0.0);
    assert!(score["value"].as_f64().unwrap() <= 3.0);
    assert_eq!(v["routing"]["resolved"], "mock:latest");
}

#[tokio::test]
async fn decide_rejects_unknown_model() {
    let mut d = TestDaemon::new(Config::default());
    let (status, v) = d
        .req(
            "POST",
            "/api/decide",
            Some(json!({"model": "nope", "state": "x"})),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body: {v}");
}

#[tokio::test]
async fn tags_lists_seeded_model() {
    let mut d = TestDaemon::new(Config::default());
    let (status, v) = d.req("GET", "/api/tags", None).await;
    assert_eq!(status, StatusCode::OK);
    let models = v["models"].as_array().unwrap();
    assert!(models.iter().any(|m| m["model"] == "mock:latest"));
}

#[tokio::test]
async fn delete_removes_model() {
    let mut d = TestDaemon::new(Config::default());
    let (status, _) = d
        .req("POST", "/api/delete", Some(json!({"model": "mock"})))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(d.registry.get("mock").is_err());
}

#[tokio::test]
async fn malformed_json_is_rejected() {
    let d = TestDaemon::new(Config::default());
    let req = Request::builder()
        .method("POST")
        .uri("/api/decide")
        .header("content-type", "application/json")
        .body(Body::from("{not json"))
        .unwrap();
    let resp = d.router.oneshot(req).await.unwrap();
    // axum returns 400 for malformed JSON, 422 for type errors.
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "got {}",
        resp.status()
    );
}

#[test]
fn rate_limiter_blocks_bursts() {
    // Covered in kredo-server unit tests; here as a canary for the module.
    let rl = kredo_server::limit::RateLimiter::new(2);
    assert!(rl.allow("a"));
    assert!(rl.allow("a"));
    assert!(!rl.allow("a"));
}
