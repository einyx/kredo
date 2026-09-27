//! Question schema, sequence layout, calibration and answers.
//!
//! A decision model answers a question set in one forward pass: the runner
//! produces raw head scores, and this crate turns them into calibrated
//! `Answer` values.

use kredo_api::{Answer, Probability, Question, QuestionKind, ScoreValue};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecisionError {
    #[error("question `{0}` is a `{1}` question but the model head is not compatible")]
    KindMismatch(String, &'static str),
    #[error(
        "this trained model answers only its built-in question set (ids: {0}); \
         it cannot answer arbitrary questions — omit `questions` to use the built-ins, \
         or use a zero-shot model (kredo:en, kredo:multilingual, nli) for custom questions"
    )]
    TrainedQuestionSet(String),
    #[error("choice question `{0}` has fewer than 2 options")]
    TooFewOptions(String),
}

/// How raw head outputs map onto questions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum HeadLayout {
    /// One logit per question, one per option slot: shape [n_heads, n_options].
    /// Used by multi-head encoder classifiers.
    MultiHead {
        /// Head index -> question id.
        heads: Vec<Head>,
    },
    /// Zero-shot pairwise scoring: every option of every question is scored
    /// as a (state, hypothesis) text pair through a standard cross-encoder
    /// (NLI or instruction classifier). The runner produces, per question,
    /// one positive probability per option; calibration happens here.
    Pairwise {
        /// Template rendered as `{prompt} {template}` with the option, e.g.
        /// `This example is {}.` — expanded per option.
        #[serde(default = "default_template")]
        template: String,
        /// Labels in the model's output space that count as "positive".
        positive_labels: Vec<String>,
        /// Label order of the model's logits.
        id2label: BTreeMap<String, String>,
        /// Temperature applied to every pairwise log (calibration). T > 1
        /// softens, T < 1 sharpens. `None` = 1 (raw model confidence).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temperature: Option<f64>,
    },
}

fn default_template() -> String {
    "{}".to_string()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Head {
    /// Question id this head answers.
    pub question: String,
    /// Labels ordered as the head outputs logits. For `score` heads this is
    /// `[min_label, max_label]`; the head emits a single value.
    pub labels: Vec<String>,
    /// ONNX output tensor name for this head. When set on every head, the
    /// runner reads each head from its own output (one forward pass, native
    /// multi-head graphs). When absent, flat logits are chunked evenly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Temperature for this head's softmax/sigmoid (calibration). T > 1
    /// softens the distribution, T < 1 sharpens it. `None` = 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// The static decision configuration of a model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionSpec {
    pub layout: HeadLayout,
}

impl DecisionSpec {
    /// Validate a question set against the head layout.
    pub fn validate(&self, questions: &[Question]) -> Result<(), DecisionError> {
        match &self.layout {
            HeadLayout::Pairwise { .. } => {
                for q in questions {
                    match q.kind {
                        QuestionKind::Choice => {
                            if q.options.len() < 2 {
                                return Err(DecisionError::TooFewOptions(q.id.clone()));
                            }
                        }
                        QuestionKind::Score => {
                            // Two anchor options bracket the scale; when
                            // absent, the [min, max] numbers are the anchors.
                            if !q.options.is_empty() && q.options.len() != 2 {
                                return Err(DecisionError::KindMismatch(
                                    q.id.clone(),
                                    "score questions need exactly two anchors (min, max)",
                                ));
                            }
                        }
                        QuestionKind::Noul => {
                            if !q.options.is_empty() && q.options.len() != 2 {
                                return Err(DecisionError::KindMismatch(
                                    q.id.clone(),
                                    "noul questions need exactly two options or none",
                                ));
                            }
                        }
                    }
                }
                Ok(())
            }
            HeadLayout::MultiHead { heads } => {
                for q in questions {
                    let Some(_head) = heads.iter().find(|h| h.question == q.id) else {
                        let supported = heads
                            .iter()
                            .map(|h| h.question.as_str())
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(DecisionError::TrainedQuestionSet(supported));
                    };
                    let head = heads.iter().find(|h| h.question == q.id).unwrap();
                    match q.kind {
                        QuestionKind::Choice => {
                            if q.options.len() < 2 {
                                return Err(DecisionError::TooFewOptions(q.id.clone()));
                            }
                            if head.labels.len() != q.options.len() {
                                return Err(DecisionError::KindMismatch(
                                    q.id.clone(),
                                    "option count does not match head labels",
                                ));
                            }
                        }
                        QuestionKind::Score | QuestionKind::Noul => {
                            if head.labels.len() != 1 && head.labels.len() != 2 {
                                return Err(DecisionError::KindMismatch(
                                    q.id.clone(),
                                    "scalar head expected",
                                ));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Expand every question/option into the hypothesis text to score
    /// pairwise. Returns `(question_index, option_index, hypothesis)`.
    pub fn hypotheses(&self, questions: &[Question]) -> Vec<(usize, usize, String)> {
        let HeadLayout::Pairwise { template, .. } = &self.layout else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (qi, q) in questions.iter().enumerate() {
            match q.kind {
                // Zero-shot `choice`: each option becomes a standalone
                // hypothesis, e.g. "This example is about a refund."
                QuestionKind::Choice => {
                    let options = if q.options.is_empty() {
                        (0..2).map(|i| i.to_string()).collect::<Vec<_>>()
                    } else {
                        q.options.clone()
                    };
                    for (oi, opt) in options.iter().enumerate() {
                        out.push((qi, oi, template.replace("{}", opt)));
                    }
                }
                // `noul`: the prompt itself is the hypothesis statement;
                // P(entailment) is the answer.
                QuestionKind::Noul => {
                    out.push((qi, 0, q.prompt.clone()));
                }
                // `score`: two anchors bracketing the scale, judged in the
                // context of the prompt.
                QuestionKind::Score => {
                    let anchors = if q.options.len() == 2 {
                        q.options.clone()
                    } else {
                        vec![
                            q.min.unwrap_or(0.0).to_string(),
                            q.max.unwrap_or(3.0).to_string(),
                        ]
                    };
                    for (oi, anchor) in anchors.iter().enumerate() {
                        out.push((qi, oi, format!("{} This is {}.", q.prompt, anchor)));
                    }
                }
            }
        }
        out
    }

    /// Calibrate pairwise positive probabilities (`raw[qi][oi]` in [0,1])
    /// into answers.
    ///
    /// `choice` questions softmax over their options; `score` questions map
    /// `(p_max - p_min)` onto the declared scale; `noul` questions report the
    /// positive option's probability.
    pub fn answer_pairwise(
        &self,
        questions: &[Question],
        raw: &[Vec<f64>],
    ) -> Result<Vec<Answer>, DecisionError> {
        let temperature = match &self.layout {
            HeadLayout::Pairwise { temperature, .. } => *temperature,
            _ => None,
        };
        let mut out = Vec::with_capacity(questions.len());
        for (qi, q) in questions.iter().enumerate() {
            let row = raw.get(qi).ok_or_else(|| {
                DecisionError::KindMismatch(q.id.clone(), "missing pairwise scores")
            })?;
            let answer = match q.kind {
                QuestionKind::Choice => {
                    let logits_for_softmax: Vec<f64> =
                        row.iter().map(|p| p.ln().max(-100.0)).collect();
                    let ps = softmax(&scale_logits(&logits_for_softmax, temperature));
                    let labels = match q.options.is_empty() {
                        true => (0..row.len()).map(|i| i.to_string()).collect::<Vec<_>>(),
                        false => q.options.clone(),
                    };
                    let probabilities = labels
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
                QuestionKind::Score => {
                    let min = q.min.unwrap_or(0.0);
                    let max = q.max.unwrap_or(3.0);
                    let p_max = row.last().copied().unwrap_or(0.0);
                    let p_min = row.first().copied().unwrap_or(0.0);
                    let unit = pairwise_unit(p_max, p_min, temperature);
                    let value = min + (max - min) * unit;
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities: Vec::new(),
                        score: Some(ScoreValue {
                            value,
                            normalized: unit,
                            min,
                            max,
                        }),
                        p: None,
                    }
                }
                QuestionKind::Noul => {
                    let p = row.first().copied().unwrap_or(0.0);
                    let p = match temperature {
                        Some(t) if t > 0.0 && (t - 1.0).abs() > 1e-9 => {
                            let z = (p.max(1e-12) / (1.0 - p).max(1e-12)).ln();
                            sigmoid(z / t)
                        }
                        _ => p,
                    };
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities: Vec::new(),
                        score: None,
                        p: Some(p),
                    }
                }
            };
            out.push(answer);
        }
        Ok(out)
    }

    /// Turn raw head outputs into calibrated answers.
    ///
    /// `raw[i][j]` is head `i`'s score for label `j`. `choice` heads are
    /// softmaxed over their labels, `score` heads are squashed through a
    /// sigmoid and mapped onto the declared [min, max] scale, `noul` heads
    /// are squashed to P(affirmative).
    pub fn answer(
        &self,
        questions: &[Question],
        raw: &[Vec<f64>],
    ) -> Result<Vec<Answer>, DecisionError> {
        let HeadLayout::MultiHead { heads } = &self.layout else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(questions.len());
        for q in questions {
            let (idx, head) = heads
                .iter()
                .enumerate()
                .find(|(_, h)| h.question == q.id)
                .ok_or_else(|| DecisionError::KindMismatch(q.id.clone(), "no matching head"))?;
            let logits = raw
                .get(idx)
                .ok_or_else(|| DecisionError::KindMismatch(q.id.clone(), "missing head output"))?;
            let answer = match q.kind {
                QuestionKind::Choice => {
                    let ps = softmax(&scale_logits(logits, head.temperature));
                    let mut probabilities = Vec::with_capacity(head.labels.len());
                    for (label, p) in head.labels.iter().zip(ps) {
                        probabilities.push(Probability {
                            label: label.clone(),
                            p,
                        });
                    }
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities,
                        score: None,
                        p: None,
                    }
                }
                QuestionKind::Score => {
                    let min = q.min.unwrap_or(0.0);
                    let max = q.max.unwrap_or(3.0);
                    let unit = if logits.len() >= 2 {
                        softmax(&scale_logits(logits, head.temperature))[1]
                    } else {
                        sigmoid(scale_logits(logits, head.temperature)[0])
                    };
                    let value = min + (max - min) * unit;
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities: Vec::new(),
                        score: Some(ScoreValue {
                            value,
                            normalized: unit,
                            min,
                            max,
                        }),
                        p: None,
                    }
                }
                QuestionKind::Noul => {
                    let p = if logits.len() >= 2 {
                        softmax(&scale_logits(logits, head.temperature))[1]
                    } else {
                        sigmoid(scale_logits(logits, head.temperature)[0])
                    };
                    Answer {
                        id: q.id.clone(),
                        kind: q.kind,
                        probabilities: Vec::new(),
                        score: None,
                        p: Some(p),
                    }
                }
            };
            out.push(answer);
        }
        Ok(out)
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Temperature-scale logits: `z / T`. `None` or T ≈ 1 returns the input
/// unchanged. This is the standard temperature-scaling calibration: it
/// preserves the argmax and the ordering while adjusting confidence.
pub fn scale_logits(logits: &[f64], temperature: Option<f64>) -> Vec<f64> {
    match temperature {
        Some(t) if t > 0.0 && (t - 1.0).abs() > 1e-9 => logits.iter().map(|z| z / t).collect(),
        _ => logits.to_vec(),
    }
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Temperature-scale a pairwise unit through logit space. At the identity
/// temperature the original `p_pos - p_min` semantics are preserved
/// unchanged; scaling only engages when a temperature is configured.
fn pairwise_unit(p_pos: f64, p_neg: f64, temperature: Option<f64>) -> f64 {
    match temperature {
        Some(t) if t > 0.0 && (t - 1.0).abs() > 1e-9 => {
            let z = (p_pos.max(1e-12) / p_neg.max(1e-12)).ln();
            sigmoid(z / t)
        }
        _ => (p_pos - p_neg).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice_q() -> Question {
        Question {
            id: "intent".into(),
            kind: QuestionKind::Choice,
            prompt: "What is the intent?".into(),
            options: vec!["refund".into(), "bug".into(), "other".into()],
            min: None,
            max: None,
        }
    }

    fn spec() -> DecisionSpec {
        DecisionSpec {
            layout: HeadLayout::MultiHead {
                heads: vec![Head {
                    question: "intent".into(),
                    labels: vec!["refund".into(), "bug".into(), "other".into()],
                    output: None,
                    temperature: None,
                }],
            },
        }
    }

    #[test]
    fn choice_softmax_sums_to_one() {
        let answers = spec()
            .answer(&[choice_q()], &[vec![1.0, 0.2, -0.5]])
            .unwrap();
        let sum: f64 = answers[0].probabilities.iter().map(|p| p.p).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert_eq!(answers[0].top().unwrap().label, "refund");
    }

    #[test]
    fn score_maps_onto_scale() {
        let q = Question {
            id: "frustration".into(),
            kind: QuestionKind::Score,
            prompt: "Frustration level?".into(),
            options: vec![],
            min: Some(0.0),
            max: Some(3.0),
        };
        let s = DecisionSpec {
            layout: HeadLayout::MultiHead {
                heads: vec![Head {
                    question: "frustration".into(),
                    labels: vec!["low".into(), "high".into()],
                    output: None,
                    temperature: None,
                }],
            },
        };
        let answers = s.answer(&[q], &[vec![0.0, 1.0]]).unwrap();
        let score = answers[0].score.as_ref().unwrap();
        assert!(score.value > 0.0 && score.value < 3.0);
        assert!((score.normalized - 1.0f64 / (1.0 + (-1.0f64).exp())).abs() < 1e-9);
    }

    #[test]
    fn validation_rejects_mismatched_options() {
        assert!(spec().validate(&[choice_q()]).is_ok());
        let bad = Question {
            options: vec!["only".into()],
            ..choice_q()
        };
        assert!(spec().validate(&[bad]).is_err());
    }

    #[test]
    fn temperature_preserves_argmax_but_changes_confidence() {
        let s = DecisionSpec {
            layout: HeadLayout::MultiHead {
                heads: vec![Head {
                    question: "intent".into(),
                    labels: vec!["refund".into(), "bug".into(), "other".into()],
                    output: None,
                    temperature: Some(2.0),
                }],
            },
        };
        let raw = &[vec![3.0, 1.0, 0.5]];
        let hot = s.answer(&[choice_q()], raw).unwrap();
        let cold = spec().answer(&[choice_q()], raw).unwrap();
        assert_eq!(hot[0].top().unwrap().label, "refund");
        let hot_p = hot[0].top().unwrap().p;
        let cold_p = cold[0].top().unwrap().p;
        assert!(hot_p < cold_p, "T=2 must soften: {hot_p} vs {cold_p}");
        let sum: f64 = hot[0].probabilities.iter().map(|p| p.p).sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn scale_logits_identity_and_sharpen() {
        assert_eq!(scale_logits(&[1.0, 2.0], None), vec![1.0, 2.0]);
        assert_eq!(scale_logits(&[1.0, 2.0], Some(1.0)), vec![1.0, 2.0]);
        assert_eq!(scale_logits(&[2.0, 4.0], Some(2.0)), vec![1.0, 2.0]);
    }

    #[test]
    fn pairwise_temperature_softens_noul() {
        let q = Question {
            id: "is_urgent".into(),
            kind: QuestionKind::Noul,
            prompt: "urgent".into(),
            options: vec![],
            min: None,
            max: None,
        };
        let layout = |t: Option<f64>| DecisionSpec {
            layout: HeadLayout::Pairwise {
                template: "{}".into(),
                positive_labels: vec!["entailment".into()],
                id2label: Default::default(),
                temperature: t,
            },
        };
        let raw = &[vec![0.9, 0.05]];
        let base = layout(None)
            .answer_pairwise(std::slice::from_ref(&q), raw)
            .unwrap()[0]
            .p
            .unwrap();
        let hot = layout(Some(3.0)).answer_pairwise(&[q], raw).unwrap()[0]
            .p
            .unwrap();
        assert!(hot < base && hot > 0.5, "softened but same side: {hot}");
    }
}
