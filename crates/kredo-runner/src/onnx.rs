//! ONNX Runtime engines.
//!
//! Both engines load the model graph from a blob and answer questions in a
//! single forward pass per input pair. fp32 on CPU; fp16 conversion happens
//! at export time for GPU builds (`--features cuda`).

use crate::{Device, Engine, RunnerError};
use kredo_api::{Answer, Question};
use kredo_registry::LocalModel;
use ndarray::{Array2, ArrayD, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Locate the graph blob and tokenizer blob of a pulled model.
struct Artifacts {
    graph: PathBuf,
    tokenizer: Tokenizer,
}

impl Artifacts {
    fn load(model: &LocalModel, registry: &kredo_registry::Registry) -> Result<Self, RunnerError> {
        let graph_entry = model
            .digests
            .iter()
            .find(|(path, _)| path.ends_with(".onnx"))
            .ok_or_else(|| RunnerError::MissingBlob("*.onnx".into()))?;
        let tok_entry = model
            .digests
            .iter()
            .find(|(path, _)| path.ends_with("tokenizer.json"))
            .ok_or_else(|| RunnerError::MissingBlob("tokenizer.json".into()))?;

        let graph = registry.blob_path(graph_entry.1);
        if !graph.exists() {
            return Err(RunnerError::MissingBlob(graph_entry.0.clone()));
        }
        let tok_path = registry.blob_path(tok_entry.1);
        if !tok_path.exists() {
            return Err(RunnerError::MissingBlob(tok_entry.0.clone()));
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| RunnerError::Other(anyhow::anyhow!("tokenizer: {e}")))?;

        let max_seq = model.manifest.max_seq_len.max(8);
        let mut tokenizer = tokenizer;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_seq,
                ..Default::default()
            }))
            .map_err(|e| RunnerError::Other(anyhow::anyhow!("truncation: {e}")))?;
        // Match the training pipeline's preprocessing (pad to max_length).
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(max_seq),
            ..Default::default()
        }));

        Ok(Self { graph, tokenizer })
    }
}

fn build_session(graph: &PathBuf, device: Device) -> Result<Session, RunnerError> {
    let builder = Session::builder()
        .map_err(|e| RunnerError::Other(e.into()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RunnerError::Other(e.into()))?
        .with_intra_threads(num_cpus())
        .map_err(|e| RunnerError::Other(e.into()))?;
    #[allow(unused_mut, unused_assignments)]
    let mut builder = builder;
    if device == Device::Cuda {
        #[cfg(feature = "cuda")]
        {
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(|e| RunnerError::Other(e.into()))?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!(
                "cuda requested but jeff was built without the `cuda` feature; using CPU"
            );
        }
    }
    builder
        .commit_from_file(graph)
        .map_err(|e| RunnerError::Other(e.into()))
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Run one (premise, hypothesis) pair; returns raw logits.
fn run_pair(
    session: &mut Session,
    tokenizer: &Tokenizer,
    premise: &str,
    hypothesis: &str,
) -> Result<Vec<f32>, RunnerError> {
    let enc = tokenizer
        .encode((premise, hypothesis), true)
        .map_err(|e| RunnerError::Other(anyhow::anyhow!("encode: {e}")))?;
    let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
    let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
    let types: Vec<i64> = enc.get_type_ids().iter().map(|&i| i as i64).collect();
    let len = ids.len();

    let inputs = ort::inputs![
        "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, len), ids.clone()).map_err(|e| RunnerError::Other(e.into()))?)
            .map_err(|e| RunnerError::Other(e.into()))?,
        "attention_mask" => Tensor::from_array(Array2::from_shape_vec((1, len), mask.clone()).map_err(|e| RunnerError::Other(e.into()))?)
            .map_err(|e| RunnerError::Other(e.into()))?,
    ];
    let has_types = session
        .inputs
        .iter()
        .any(|i| i.name.contains("token_type_ids"));
    let outputs = if has_types {
        session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(Array2::<i64>::from_shape_vec((1, len), types.clone()).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
                "attention_mask" => Tensor::from_array(Array2::<i64>::from_shape_vec((1, len), mask.clone()).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
                "token_type_ids" => Tensor::from_array(Array2::<i64>::from_shape_vec((1, len), types).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
            ])
            .map_err(|e| RunnerError::Other(e.into()))?
    } else {
        session
            .run(inputs)
            .map_err(|e| RunnerError::Other(e.into()))?
    };

    let logits = outputs["logits"]
        .try_extract_array()
        .map_err(|e| RunnerError::Other(e.into()))?;
    Ok(logits.iter().copied().collect::<Vec<f32>>())
}

/// Run a single sequence; returns raw logits (all heads flattened).
fn run_single(
    session: &mut Session,
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<Vec<f32>, RunnerError> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| RunnerError::Other(anyhow::anyhow!("encode: {e}")))?;
    let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
    let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
    let len = ids.len();
    let outputs = session
        .run(ort::inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, len), ids).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
            "attention_mask" => Tensor::from_array(Array2::from_shape_vec((1, len), mask).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
        ])
        .map_err(|e| RunnerError::Other(e.into()))?;
    let logits = outputs["logits"]
        .try_extract_array()
        .map_err(|e| RunnerError::Other(e.into()))?;
    Ok(logits.iter().copied().collect::<Vec<f32>>())
}

fn logits_from_flat(flat: &[f32]) -> Vec<f64> {
    flat.iter().map(|&v| v as f64).collect()
}

// ---------------------------------------------------------------------------
// Pairwise (zero-shot) engine
// ---------------------------------------------------------------------------

pub struct PairwiseEngine {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    spec: kredo_decision::DecisionSpec,
}

impl PairwiseEngine {
    pub fn load(
        model: &LocalModel,
        registry: &kredo_registry::Registry,
        device: Device,
    ) -> Result<Self, RunnerError> {
        let art = Artifacts::load(model, registry)?;
        let session = build_session(&art.graph, device)?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer: art.tokenizer,
            spec: model.manifest.decision.clone(),
        })
    }

    /// P(positive) from NLI-style logits.
    fn positive(&self, logits: &[f64]) -> f64 {
        let kredo_decision::HeadLayout::Pairwise {
            positive_labels,
            id2label,
            ..
        } = &self.spec.layout
        else {
            return kredo_decision::sigmoid(0.0);
        };
        let labels: Vec<&str> = (0..logits.len())
            .map(|i| {
                id2label
                    .get(&i.to_string())
                    .map(|s| s.as_str())
                    .unwrap_or("unknown")
            })
            .collect();
        let ps = kredo_decision::softmax(logits);
        if std::env::var("KREDO_DEBUG_LOGITS").is_ok() {
            eprintln!("logits={logits:?} labels={labels:?} ps={ps:?} positive={positive_labels:?}");
        }
        positive_labels
            .iter()
            .map(|pl| {
                labels
                    .iter()
                    .zip(&ps)
                    .filter(|(l, _)| **l == pl.as_str())
                    .map(|(_, p)| *p)
                    .sum::<f64>()
            })
            .sum()
    }
}

impl Engine for PairwiseEngine {
    fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, RunnerError> {
        self.spec
            .validate(questions)
            .map_err(|e| RunnerError::Other(e.into()))?;
        let hyps = self.spec.hypotheses(questions);
        let mut raw: Vec<Vec<f64>> = vec![Vec::new(); questions.len()];
        {
            let mut session = self
                .session
                .lock()
                .map_err(|_| RunnerError::Other(anyhow::anyhow!("session lock poisoned")))?;
            for (qi, _, hypothesis) in &hyps {
                let logits = run_pair(&mut session, &self.tokenizer, state, hypothesis)?;
                let p = self.positive(&logits_from_flat(&logits));
                raw[*qi].push(p);
            }
        }
        self.spec
            .answer_pairwise(questions, &raw)
            .map_err(|e| RunnerError::Other(e.into()))
    }

    fn kind(&self) -> &'static str {
        "onnx-pairwise"
    }
}

// ---------------------------------------------------------------------------
// Multi-head engine
// ---------------------------------------------------------------------------

pub struct MultiHeadEngine {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    spec: kredo_decision::DecisionSpec,
}

impl MultiHeadEngine {
    pub fn load(
        model: &LocalModel,
        registry: &kredo_registry::Registry,
        device: Device,
    ) -> Result<Self, RunnerError> {
        let art = Artifacts::load(model, registry)?;
        let session = build_session(&art.graph, device)?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer: art.tokenizer,
            spec: model.manifest.decision.clone(),
        })
    }
}

impl Engine for MultiHeadEngine {
    fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, RunnerError> {
        self.spec
            .validate(questions)
            .map_err(|e| RunnerError::Other(e.into()))?;
        let kredo_decision::HeadLayout::MultiHead { heads } = &self.spec.layout else {
            return Ok(Vec::new());
        };
        let (all_flat, per_head_outputs) = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| RunnerError::Other(anyhow::anyhow!("session lock poisoned")))?;
            if heads.iter().all(|h| h.output.is_some()) {
                // Native multi-head graph: one output tensor per head.
                let enc = self
                    .tokenizer
                    .encode(state, true)
                    .map_err(|e| RunnerError::Other(anyhow::anyhow!("encode: {e}")))?;
                let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
                let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
                let len = ids.len();
                let outputs = session
                    .run(ort::inputs![
                        "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, len), ids).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
                        "attention_mask" => Tensor::from_array(Array2::from_shape_vec((1, len), mask).map_err(|e| RunnerError::Other(e.into()))?).map_err(|e| RunnerError::Other(e.into()))?,
                    ])
                    .map_err(|e| RunnerError::Other(e.into()))?;
                let mut per_head: Vec<Vec<f64>> = Vec::with_capacity(heads.len());
                for h in heads {
                    let name = h.output.as_deref().expect("checked above");
                    let t = outputs[name]
                        .try_extract_array()
                        .map_err(|e| RunnerError::Other(e.into()))?;
                    per_head.push(t.iter().map(|&v: &f32| v as f64).collect());
                }
                (Vec::new(), per_head)
            } else {
                let flat = run_single(&mut session, &self.tokenizer, state)?;
                (flat, Vec::new())
            }
        };
        let raw: Vec<Vec<f64>> = if !per_head_outputs.is_empty() {
            per_head_outputs
        } else {
            let values = logits_from_flat(&all_flat);
            // Accept [1, H*O], [H*O] or [H, O] flat layouts.
            let per_head = values.len() / heads.len().max(1);
            values
                .chunks(per_head.max(1))
                .take(heads.len())
                .map(|c| c.to_vec())
                .collect()
        };
        self.spec
            .answer(questions, &raw)
            .map_err(|e| RunnerError::Other(e.into()))
    }

    fn kind(&self) -> &'static str {
        "onnx-multihead"
    }
}

// Keep unused-import warnings away on cfg paths.
#[allow(dead_code)]
fn _shape_helper(v: ArrayD<f32>) -> ArrayD<f32> {
    let _ = IxDyn(&[1]);
    let _: BTreeMap<String, String> = BTreeMap::new();
    v
}
