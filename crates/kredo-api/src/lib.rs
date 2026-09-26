//! Wire types for jeff: the TypeSafe-compatible `/v1/systemone` contract and
//! jeff's native `/api/*` surface. See docs/api.md.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_PORT: u16 = 21435;

// ---------------------------------------------------------------------------
// Questions
// ---------------------------------------------------------------------------

/// Typed question kinds supported by decision models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    /// Pick among labeled options; returns calibrated probabilities.
    Choice,
    /// Regression on a bounded numeric scale; returns a score and normalized value.
    Score,
    /// Binary "no / uncertain / label" judgment; returns P(not-null).
    Noul,
}

/// A typed question posed against the state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// Stable identifier for the answer.
    pub id: String,
    /// Question kind: `choice`, `score` or `noul`.
    #[serde(rename = "type")]
    pub kind: QuestionKind,
    /// The question text, e.g. "What is the user's intent?".
    pub prompt: String,
    /// Options for `choice` questions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    /// Minimum of the score scale (`score` only). Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Maximum of the score scale (`score` only). Default 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
}

// ---------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------

/// One calibrated probability over a label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probability {
    pub label: String,
    pub p: f64,
}

/// A `score` answer: value on the declared scale plus normalization to [0,1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreValue {
    pub value: f64,
    pub normalized: f64,
    pub min: f64,
    pub max: f64,
}

/// The answer to one question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: QuestionKind,
    /// Calibrated distribution (`choice` only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probabilities: Vec<Probability>,
    /// Regressed value (`score` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<ScoreValue>,
    /// P(affirmative) for `noul` questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p: Option<f64>,
}

impl Answer {
    /// Highest-probability label of a `choice` answer.
    pub fn top(&self) -> Option<&Probability> {
        self.probabilities.iter().max_by(|a, b| a.p.total_cmp(&b.p))
    }
}

// ---------------------------------------------------------------------------
// Requests / responses
// ---------------------------------------------------------------------------

/// `POST /v1/systemone` — TypeSafe-compatible decision request.
///
/// `state` may be a plain string or arbitrary JSON (stringified into the
/// model's text view). `questions` may be omitted when the model ships a
/// built-in question set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub state: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub questions: Vec<Question>,
    /// Optional model tag; defaults to the server default model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// `POST /v1/systemone` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: Vec<Answer>,
    /// Total inference time in milliseconds.
    pub elapsed_ms: f64,
}

/// `POST /v1/decisions` — batch of states against one question set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionsRequest {
    pub model: Option<String>,
    pub states: Vec<serde_json::Value>,
    pub questions: Vec<Question>,
}

/// `POST /api/decide` — jeff-native decide with routing + timings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideRequest {
    pub model: String,
    pub state: serde_json::Value,
    #[serde(default)]
    pub questions: Vec<Question>,
}

/// Routing information reported by `/api/decide`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Routing {
    /// The concrete tag the router resolved to, e.g. `kredo:en`.
    pub resolved: String,
    /// Detected script (`latn`, `cyrl`, `hani`, ...), when routed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// Detected language hint, when routed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Router overhead in milliseconds.
    pub route_ms: f64,
}

/// `POST /api/decide` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideResponse {
    pub model: String,
    pub answers: Vec<Answer>,
    pub routing: Routing,
    pub load_ms: f64,
    pub infer_ms: f64,
}

// ---------------------------------------------------------------------------
// Model listing
// ---------------------------------------------------------------------------

/// One entry of `GET /v1/models` (TypeSafe-compatible shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub questions: Vec<Question>,
}

/// `GET /api/tags` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagsResponse {
    pub models: Vec<ModelInfo>,
}

/// `GET /api/ps` entry: a model currently resident.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub expires_at: String,
}

/// `GET /api/ps` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsResponse {
    pub models: Vec<RunningModel>,
}

/// `POST /api/pull` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub model: String,
    #[serde(default)]
    pub insecure: bool,
}

/// NDJSON progress event streamed by `/api/pull`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgress {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

/// `POST /api/show` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowRequest {
    pub model: String,
}

/// `GET /api/show` / `POST /api/show` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowResponse {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub details: BTreeMap<String, serde_json::Value>,
    pub questions: Vec<Question>,
}

/// Error envelope used across the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.error)
    }
}

impl ApiError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { error: msg.into() }
    }
}

/// Render the state into the flat text view a decision model reads.
pub fn state_to_text(state: &serde_json::Value) -> String {
    match state {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
