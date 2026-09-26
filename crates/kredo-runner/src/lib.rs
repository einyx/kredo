//! Inference engines for decision models.
//!
//! [`Engine`] is the contract the scheduler holds; [`load`] builds one from a
//! pulled model. Two ONNX engines ship: `onnx-pairwise` (zero-shot scoring of
//! every option through a cross-encoder) and `onnx-multihead` (single forward
//! pass over dedicated heads). A deterministic [`MockEngine`] backs tests.

pub mod mock;
pub mod onnx;
pub mod verification;

use kredo_api::{Answer, Question};
use kredo_registry::LocalModel;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("unsupported engine `{0}`")]
    UnsupportedEngine(String),
    #[error("model is missing blob for `{0}`")]
    MissingBlob(String),
    #[error("runner: {0}")]
    Other(#[from] anyhow::Error),
}

/// Execution device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn from_env() -> Self {
        match std::env::var("KREDO_DEVICE").as_deref() {
            Ok("cuda") => Device::Cuda,
            _ => Device::Cpu,
        }
    }
}

/// A loaded decision model, ready to answer question sets.
pub trait Engine: Send + Sync {
    /// Answer `questions` against `state` in one pass (per engine semantics).
    fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, RunnerError>;

    /// Short engine identifier, e.g. `onnx-pairwise`.
    fn kind(&self) -> &'static str;
}

/// Load an engine for a pulled model.
pub fn load(
    model: &LocalModel,
    registry: &kredo_registry::Registry,
    device: Device,
) -> Result<Box<dyn Engine>, RunnerError> {
    match model.manifest.engine.as_str() {
        // Deterministic engine for tests and smoke runs; no files needed.
        "mock" => Ok(Box::new(mock::MockEngine::new(model.manifest.full_name()))),
        "onnx-pairwise" => Ok(Box::new(onnx::PairwiseEngine::load(
            model, registry, device,
        )?)),
        "onnx-multihead" => Ok(Box::new(onnx::MultiHeadEngine::load(
            model, registry, device,
        )?)),
        other => Err(RunnerError::UnsupportedEngine(other.into())),
    }
}
