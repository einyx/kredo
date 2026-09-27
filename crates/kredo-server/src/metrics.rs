//! Hand-rolled Prometheus metrics: counters and latency histograms with
//! fixed buckets, exported as text at `GET /metrics`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

pub struct Histogram {
    buckets: Vec<AtomicU64>,
    sum_nanos: AtomicU64,
    count: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: LATENCY_BUCKETS.iter().map(|_| AtomicU64::new(0)).collect(),
            sum_nanos: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, d: Duration) {
        let secs = d.as_secs_f64();
        for (i, b) in LATENCY_BUCKETS.iter().enumerate() {
            if secs <= *b {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.sum_nanos
            .fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self, name: &str, help: &str, out: &mut String) {
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} histogram\n"));
        let mut cumulative = 0u64;
        for (i, b) in LATENCY_BUCKETS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{b}\"}} {cumulative}\n"));
        }
        out.push_str(&format!(
            "{name}_bucket{{le=\"+Inf\"}} {}\n",
            self.count.load(Ordering::Relaxed)
        ));
        let sum = self.sum_nanos.load(Ordering::Relaxed) as f64 / 1e9;
        out.push_str(&format!("{name}_sum {sum}\n"));
        out.push_str(&format!(
            "{name}_count {}\n",
            self.count.load(Ordering::Relaxed)
        ));
    }
}

#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub rate_limited_total: AtomicU64,
    pub models_loaded: AtomicU64,
    pub decisions_total: AtomicU64,
    pub request_duration: Histogram,
    #[allow(private_interfaces)]
    pub inference_duration: Histogram,
    /// Decisions and inference latency broken down by resolved model tag.
    pub by_model: Mutex<BTreeMap<String, ModelStats>>,
    /// Shadow-model evaluation counters.
    pub shadow_total: AtomicU64,
    pub shadow_agree_total: AtomicU64,
    #[allow(private_interfaces)]
    pub shadow_duration: Histogram,
}

/// Per-model decision counters and latency.
#[derive(Default)]
pub struct ModelStats {
    pub decisions: u64,
    pub duration: Histogram,
}

impl Metrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a completed decision for `model`.
    pub fn record_decision(&self, model: &str, duration: Duration) {
        self.decisions_total.fetch_add(1, Ordering::Relaxed);
        self.inference_duration.observe(duration);
        let mut map = self.by_model.lock().unwrap();
        let stats = map.entry(model.to_string()).or_default();
        stats.decisions += 1;
        stats.duration.observe(duration);
    }

    /// Record a shadow-model evaluation.
    pub fn record_shadow(&self, agree: bool, duration: Duration) {
        self.shadow_total.fetch_add(1, Ordering::Relaxed);
        if agree {
            self.shadow_agree_total.fetch_add(1, Ordering::Relaxed);
        }
        self.shadow_duration.observe(duration);
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        for (name, help, v) in [
            (
                "jeff_requests_total",
                "Total HTTP requests.",
                self.requests_total.load(Ordering::Relaxed),
            ),
            (
                "jeff_errors_total",
                "Total HTTP 5xx responses.",
                self.errors_total.load(Ordering::Relaxed),
            ),
            (
                "jeff_rate_limited_total",
                "Requests rejected by the rate limiter.",
                self.rate_limited_total.load(Ordering::Relaxed),
            ),
            (
                "kredo_decisions_total",
                "Total decisions computed.",
                self.decisions_total.load(Ordering::Relaxed),
            ),
        ] {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"
            ));
        }
        out.push_str(&format!(
            "# HELP jeff_models_loaded Models currently resident.\n# TYPE jeff_models_loaded gauge\njeff_models_loaded {}\n",
            self.models_loaded.load(Ordering::Relaxed)
        ));
        self.request_duration.render(
            "jeff_request_duration_seconds",
            "HTTP request latency.",
            &mut out,
        );
        self.inference_duration.render(
            "jeff_inference_duration_seconds",
            "Decision inference latency (incl. queueing).",
            &mut out,
        );
        {
            let map = self.by_model.lock().unwrap();
            if !map.is_empty() {
                out.push_str("# HELP kredo_decisions_by_model_total Decisions per model.\n# TYPE kredo_decisions_by_model_total counter\n");
                for (model, stats) in map.iter() {
                    out.push_str(&format!(
                        "kredo_decisions_by_model_total{{model=\"{model}\"}} {}\n",
                        stats.decisions
                    ));
                }
            }
            for (model, stats) in map.iter() {
                stats.duration.render_labeled(
                    "kredo_model_inference_duration_seconds",
                    &format!("model=\"{model}\""),
                    "Decision inference latency.",
                    &mut out,
                );
            }
        }
        self.shadow_duration.render(
            "kredo_shadow_duration_seconds",
            "Shadow-model inference latency.",
            &mut out,
        );
        out.push_str(&format!(
            "# HELP kredo_shadow_total Shadow evaluations.\n# TYPE kredo_shadow_total counter\nkredo_shadow_total {}\n",
            self.shadow_total.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "# HELP kredo_shadow_agree_total Shadow evaluations agreeing with the served model.\n# TYPE kredo_shadow_agree_total counter\nkredo_shadow_agree_total {}\n",
            self.shadow_agree_total.load(Ordering::Relaxed)
        ));
        out
    }
}

use std::collections::BTreeMap;
use std::sync::Mutex;

impl Histogram {
    /// Render a histogram with an extra constant label set.
    fn render_labeled(&self, name: &str, labels: &str, help: &str, out: &mut String) {
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} histogram\n"));
        let mut cumulative = 0u64;
        for (i, b) in LATENCY_BUCKETS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{{labels},le=\"{b}\"}} {cumulative}\n"
            ));
        }
        out.push_str(&format!(
            "{name}_bucket{{{labels},le=\"+Inf\"}} {}\n",
            self.count.load(Ordering::Relaxed)
        ));
        let sum = self.sum_nanos.load(Ordering::Relaxed) as f64 / 1e9;
        out.push_str(&format!("{name}_sum{{{labels}}} {sum}\n"));
        out.push_str(&format!(
            "{name}_count{{{labels}}} {}\n",
            self.count.load(Ordering::Relaxed)
        ));
    }
}
