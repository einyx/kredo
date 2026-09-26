//! Deterministic mock engine used by tests and integration suites.
//!
//! Scores derive from a stable hash of (question id, option, state), so the
//! same input always yields the same decision — useful for golden fixtures.

use crate::{Engine, RunnerError};
use kredo_api::{Answer, Probability, Question, ScoreValue};
use sha2::{Digest, Sha256};

pub struct MockEngine {
    pub name: String,
}

impl MockEngine {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    fn score(label: &str, state: &str) -> f64 {
        let mut h = Sha256::new();
        h.update(label.as_bytes());
        h.update(b"\x00");
        h.update(state.as_bytes());
        let d = h.finalize();
        // Map to [-2, 2] for stable logit-ish values.
        (d[0] as f64 / 255.0) * 4.0 - 2.0
    }
}

impl Engine for MockEngine {
    fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, RunnerError> {
        let mut out = Vec::with_capacity(questions.len());
        for q in questions {
            let answer = match q.kind {
                kredo_api::QuestionKind::Choice => {
                    let logits: Vec<f64> = q
                        .options
                        .iter()
                        .map(|o| Self::score(&format!("{}:{}", q.id, o), state))
                        .collect();
                    let ps = kredo_decision::softmax(&logits);
                    let probabilities = q
                        .options
                        .iter()
                        .zip(ps)
                        .map(|(l, p)| Probability {
                            label: l.clone(),
                            p,
                        })
                        .collect();
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities,
                        score: None,
                        p: None,
                    }
                }
                kredo_api::QuestionKind::Score => {
                    let min = q.min.unwrap_or(0.0);
                    let max = q.max.unwrap_or(3.0);
                    let unit = (Self::score(&q.id, state) + 2.0) / 4.0;
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities: vec![],
                        score: Some(ScoreValue {
                            value: min + (max - min) * unit,
                            normalized: unit,
                            min,
                            max,
                        }),
                        p: None,
                    }
                }
                kredo_api::QuestionKind::Noul => {
                    let unit = (Self::score(&q.id, state) + 2.0) / 4.0;
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities: vec![],
                        score: None,
                        p: Some(unit),
                    }
                }
            };
            out.push(answer);
        }
        Ok(out)
    }

    fn kind(&self) -> &'static str {
        "mock"
    }
}
