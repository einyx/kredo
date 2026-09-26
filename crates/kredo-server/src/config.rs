//! Server configuration, loaded from environment with optional CLI overrides.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Bind address (or client base URL when set by clients).
    pub host: SocketAddr,
    /// Model store root. None = default (~/.kredo/models or $KREDO_MODELS).
    pub models_dir: Option<PathBuf>,
    /// Seconds a loaded model stays resident without use.
    pub keep_alive: Duration,
    /// Bearer token; when set, every request must present it.
    pub api_key: Option<String>,
    /// Max concurrent inferences per model.
    pub max_concurrency: usize,
    /// Per-request timeout.
    pub request_timeout: Duration,
    /// Requests per minute per client (0 = unlimited).
    pub rate_limit_per_min: u32,
    /// TLS certificate / key for networked mode.
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    /// Log as JSON (machine) instead of text (human).
    pub log_format: LogFormat,
    /// Max request body bytes.
    pub body_limit: usize,
    /// Allow cross-origin requests (dev/browser use).
    pub cors: bool,
    /// Refuse to load models without a passing verification block.
    pub require_verified: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: SocketAddr::from(([127, 0, 0, 1], kredo_api::DEFAULT_PORT)),
            models_dir: None,
            keep_alive: crate::DEFAULT_KEEP_ALIVE,
            api_key: None,
            max_concurrency: 4,
            request_timeout: Duration::from_secs(30),
            rate_limit_per_min: 0,
            tls_cert: None,
            tls_key: None,
            log_format: LogFormat::Text,
            body_limit: 1 << 20,
            cors: false,
            require_verified: false,
        }
    }
}

impl Config {
    /// Load from the KREDO_* environment variables.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Ok(h) = std::env::var("KREDO_HOST") {
            if let Ok(addr) = h.parse() {
                c.host = addr;
            }
        }
        if let Ok(d) = std::env::var("KREDO_MODELS") {
            c.models_dir = Some(PathBuf::from(d));
        }
        if let Ok(s) = std::env::var("KREDO_KEEP_ALIVE") {
            if let Ok(secs) = s.parse::<u64>() {
                c.keep_alive = Duration::from_secs(secs);
            }
        }
        c.api_key = std::env::var("KREDO_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        if let Ok(s) = std::env::var("KREDO_MAX_CONCURRENCY") {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 {
                    c.max_concurrency = n;
                }
            }
        }
        if let Ok(s) = std::env::var("KREDO_REQUEST_TIMEOUT") {
            if let Ok(secs) = s.parse::<u64>() {
                c.request_timeout = Duration::from_secs(secs);
            }
        }
        if let Ok(s) = std::env::var("KREDO_RATE_LIMIT") {
            if let Ok(n) = s.parse::<u32>() {
                c.rate_limit_per_min = n;
            }
        }
        if let Ok(p) = std::env::var("KREDO_TLS_CERT") {
            c.tls_cert = Some(PathBuf::from(p));
        }
        if let Ok(p) = std::env::var("KREDO_TLS_KEY") {
            c.tls_key = Some(PathBuf::from(p));
        }
        if std::env::var("KREDO_LOG_FORMAT").as_deref() == Ok("json") {
            c.log_format = LogFormat::Json;
        }
        if std::env::var("KREDO_CORS").as_deref() == Ok("1") {
            c.cors = true;
        }
        if std::env::var("KREDO_REQUIRE_VERIFIED").as_deref() == Ok("1") {
            c.require_verified = true;
        }
        if let Ok(s) = std::env::var("KREDO_BODY_LIMIT") {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 {
                    c.body_limit = n;
                }
            }
        }
        c
    }

    /// Validate the combination of settings; returns a human-readable error.
    pub fn validate(&self) -> Result<(), String> {
        if (self.tls_cert.is_some()) != (self.tls_key.is_some()) {
            return Err("TLS requires both KREDO_TLS_CERT and KREDO_TLS_KEY".into());
        }
        if self.max_concurrency > 1024 {
            return Err("KREDO_MAX_CONCURRENCY too large (max 1024)".into());
        }
        Ok(())
    }

    /// True when bound beyond loopback.
    pub fn is_network_exposed(&self) -> bool {
        !self.host.ip().is_loopback()
    }
}
