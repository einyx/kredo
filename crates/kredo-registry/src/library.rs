//! The embedded model library: the default set of models `kredo pull` knows.
//!
//! kredo publishes small ONNX graphs and reads weight/tokenizer files from
//! their authors' Hugging Face repositories, pinned to a revision. Digests
//! are recorded at first pull when not pinned here.

use crate::Manifest;
use kredo_decision::HeadLayout;

/// Look a model up by `name` and `tag` in the embedded library.
pub fn lookup(name: &str, tag: &str) -> Option<Manifest> {
    library()
        .into_iter()
        .find(|m| m.name == name && m.tag == tag)
}

/// All models in the embedded library.
pub fn all() -> Vec<Manifest> {
    library()
}

fn pairwise_nli(id2label: &[(&str, &str)]) -> HeadLayout {
    HeadLayout::Pairwise {
        template: "This example is about {}.".into(),
        positive_labels: vec!["entailment".into()],
        id2label: id2label
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

/// Placeholder decision spec for router manifests; routers never run
/// inference themselves.
fn placeholder_spec() -> kredo_decision::DecisionSpec {
    kredo_decision::DecisionSpec {
        layout: HeadLayout::Pairwise {
            template: String::new(),
            positive_labels: vec![],
            id2label: Default::default(),
        },
    }
}

fn hf(repo: &str, files: &[(&str, Option<&str>)]) -> crate::Source {
    crate::Source {
        repo: Some(repo.into()),
        revision: "main".into(),
        files: files
            .iter()
            .map(|(path, sha)| crate::FileSpec {
                path: path.to_string(),
                url: None,
                sha256: sha.map(|s| s.to_string()),
                required: true,
            })
            .collect(),
    }
}

impl crate::Source {
    /// Override the pinned revision (chain on `hf`).
    fn with_revision(mut self, revision: &str) -> Self {
        self.revision = revision.into();
        self
    }
}

/// Pin a source to an exact upstream commit with verified digests.
fn pinned(repo: &str, revision: &str, files: &[(&str, &str)]) -> crate::Source {
    crate::Source {
        repo: Some(repo.into()),
        revision: revision.into(),
        files: files
            .iter()
            .map(|(path, sha)| crate::FileSpec {
                path: path.to_string(),
                url: None,
                sha256: Some(sha.to_string()),
                required: true,
            })
            .collect(),
    }
}

fn library() -> Vec<Manifest> {
    use kredo_api::Question;
    use kredo_api::QuestionKind::{Choice, Noul, Score};

    let jeff_questions = |lang: &str| -> Vec<Question> {
        vec![
            Question {
                id: "intent".into(),
                kind: Choice,
                prompt: format!("What does the user want in this {lang} message?"),
                options: vec![
                    "a refund".into(),
                    "a bug report".into(),
                    "a question about billing".into(),
                    "cancelling the account".into(),
                    "praise".into(),
                ],
                min: None,
                max: None,
            },
            Question {
                id: "is_urgent".into(),
                kind: Noul,
                prompt: "This message is urgent and needs a same-day response".into(),
                options: vec![],
                min: None,
                max: None,
            },
            Question {
                id: "frustration".into(),
                kind: Score,
                prompt: "How frustrated is the user?".into(),
                options: vec![
                    "not frustrated at all".into(),
                    "extremely frustrated".into(),
                ],
                min: Some(0.0),
                max: Some(3.0),
            },
            Question {
                id: "refund_requested".into(),
                kind: Noul,
                prompt: "The user explicitly asks for a refund".into(),
                options: vec![],
                min: None,
                max: None,
            },
            Question {
                id: "churn_risk".into(),
                kind: Noul,
                prompt: "The user is at risk of cancelling their subscription".into(),
                options: vec![],
                min: None,
                max: None,
            },
        ]
    };

    vec![
        // Router: delegates to a concrete tag by script/language.
        Manifest {
            schema: 1,
            name: "kredo".into(),
            tag: "triage".into(),
            description: "Trained triage decision model (BERT-tiny multi-head, 17 MB, offline)"
                .into(),
            engine: "onnx-multihead".into(),
            precision: "fp32".into(),
            max_seq_len: 128,
            source: pinned(
                "einyx/kredo-artifacts",
                "v0.1.0",
                &[
                    (
                        "model.onnx",
                        "63e9ebe813972391f6bb0bfa8a9d073310d338185c7067e35925f1e5bccdb27d",
                    ),
                    (
                        "tokenizer.json",
                        "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
                    ),
                    (
                        "verification.json",
                        "80ba691f45191376f85d319d4bc3fd7bd973f06a3b9312fd4a7e6d9da1615939",
                    ),
                ],
            ),
            decision: kredo_decision::DecisionSpec {
                layout: HeadLayout::MultiHead {
                    heads: vec![
                        kredo_decision::Head {
                            question: "intent".into(),
                            labels: vec![
                                "a refund".into(),
                                "a bug report".into(),
                                "a question about billing".into(),
                                "cancelling the account".into(),
                                "praise".into(),
                            ],
                            output: Some("head_intent".into()),
                        },
                        kredo_decision::Head {
                            question: "is_urgent".into(),
                            labels: vec!["no".into(), "yes".into()],
                            output: Some("head_is_urgent".into()),
                        },
                        kredo_decision::Head {
                            question: "frustration".into(),
                            labels: vec!["low".into(), "high".into()],
                            output: Some("head_frustration".into()),
                        },
                        kredo_decision::Head {
                            question: "refund_requested".into(),
                            labels: vec!["no".into(), "yes".into()],
                            output: Some("head_refund_requested".into()),
                        },
                        kredo_decision::Head {
                            question: "churn_risk".into(),
                            labels: vec!["no".into(), "yes".into()],
                            output: Some("head_churn_risk".into()),
                        },
                    ],
                },
            },
            questions: jeff_questions("English"),
            router: None,
            provenance: Some(crate::Provenance {
                dataset: "synthetic triage corpus (seeded templates + noise)".into(),
                seed: Some(7),
                pipeline: Some("convert v0.1.0".into()),
                metrics: [
                    ("intent.accuracy", 1.0),
                    ("is_urgent.accuracy", 1.0),
                    ("frustration.mae", 0.3434),
                    ("refund_requested.accuracy", 1.0),
                    ("churn_risk.accuracy", 0.8438),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            }),
            verification: Some(crate::Verification {
                fixture_digest: "80ba691f45191376f85d319d4bc3fd7bd973f06a3b9312fd4a7e6d9da1615939".into(),
                tolerance: 1e-4,
                verified_at: "2026-09-26T19:56:18+00:00".into(),
                cases: 3,
            }),
        },
        Manifest {
            schema: 1,
            name: "kredo".into(),
            tag: "support".into(),
            description: "Trained support-ticket routing model on real tickets (BERT-tiny multi-head, 17 MB, offline)".into(),
            engine: "onnx-multihead".into(),
            precision: "fp32".into(),
            max_seq_len: 128,
            source: pinned(
                "einyx/kredo-artifacts",
                "v0.1.0",
                &[
                    (
                        "model.onnx",
                        "50b4b393ba7965e9ee32d58b8703554060d0fcc86845b05534c5d8e066912639",
                    ),
                    (
                        "tokenizer.json",
                        "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
                    ),
                    (
                        "verification.json",
                        "2d8b388bea4094f452aa95f850b968d09276865aac743646c8750db170083a2c",
                    ),
                ],
            ),
            decision: kredo_decision::DecisionSpec {
                layout: HeadLayout::MultiHead {
                    heads: vec![
                        kredo_decision::Head {
                            question: "queue".into(),
                            labels: vec![
                                "Billing and Payments".into(),
                                "Customer Service".into(),
                                "General Inquiry".into(),
                                "Human Resources".into(),
                                "IT Support".into(),
                                "Product Support".into(),
                                "Returns and Exchanges".into(),
                                "Sales and Pre-Sales".into(),
                                "Service Outages and Maintenance".into(),
                                "Technical Support".into(),
                            ],
                            output: Some("head_queue".into()),
                        },
                        kredo_decision::Head {
                            question: "is_urgent".into(),
                            labels: vec!["no".into(), "yes".into()],
                            output: Some("head_is_urgent".into()),
                        },
                    ],
                },
            },
            questions: vec![
                kredo_api::Question {
                    id: "queue".into(),
                    kind: kredo_api::QuestionKind::Choice,
                    prompt: "Which support queue should handle this ticket?".into(),
                    options: vec![
                        "Billing and Payments".into(),
                        "Customer Service".into(),
                        "General Inquiry".into(),
                        "Human Resources".into(),
                        "IT Support".into(),
                        "Product Support".into(),
                        "Returns and Exchanges".into(),
                        "Sales and Pre-Sales".into(),
                        "Service Outages and Maintenance".into(),
                        "Technical Support".into(),
                    ],
                    min: None,
                    max: None,
                },
                kredo_api::Question {
                    id: "is_urgent".into(),
                    kind: kredo_api::QuestionKind::Noul,
                    prompt: "This ticket is high priority and needs an immediate response".into(),
                    options: vec![],
                    min: None,
                    max: None,
                },
            ],
            router: None,
            provenance: Some(crate::Provenance {
                dataset: "Tobi-Bueck/customer-support-tickets (English subset, 15k rows)".into(),
                seed: Some(7),
                pipeline: Some("convert v0.1.0".into()),
                metrics: [
                    ("queue.accuracy", 0.7109),
                    ("is_urgent.accuracy", 0.7969),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            }),
            verification: Some(crate::Verification {
                fixture_digest: "b5649aecd282be36e77fba1449b7b5046695ffed8188b07a6b6fd41b77c8bf2f".into(),
                tolerance: 1e-4,
                verified_at: "2026-09-26T21:15:00+00:00".into(),
                cases: 3,
            }),
        },
        Manifest {
            schema: 1,
            name: "kredo".into(),
            tag: "latest".into(),
            description: "Router: picks kredo:en or kredo:multilingual by language".into(),
            engine: "router".into(),
            precision: "none".into(),
            max_seq_len: 0,
            source: hf("", &[]),
            decision: placeholder_spec(),
            questions: vec![],
            router: Some(crate::Router {
                en: "kredo:en".into(),
                multilingual: "kredo:multilingual".into(),
            }),
            provenance: None,
            verification: None,
        },
        Manifest {
            schema: 1,
            name: "kredo".into(),
            tag: "en".into(),
            description: "English decision model (DistilBERT MNLI cross-encoder)".into(),
            engine: "onnx-pairwise".into(),
            precision: "fp32".into(),
            max_seq_len: 512,
            source: pinned(
                "Xenova/distilbert-base-uncased-mnli",
                "fddd480db7392a87114a6813c6acb5ede13ff4ee",
                &[
                    (
                        "onnx/model.onnx",
                        "145d2bb1a82d30d63d310172d84313963c79efb843c813d7e40af425f4ac6cc6",
                    ),
                    (
                        "tokenizer.json",
                        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                    ),
                ],
            ),
            decision: kredo_decision::DecisionSpec {
                layout: pairwise_nli(&[
                    ("0", "entailment"),
                    ("1", "neutral"),
                    ("2", "contradiction"),
                ]),
            },
            questions: jeff_questions("English"),
            router: None,
            provenance: None,
            verification: None,
        },
        Manifest {
            schema: 1,
            name: "kredo".into(),
            tag: "multilingual".into(),
            description: "100+ language decision model (mDeBERTa XNLI cross-encoder)".into(),
            engine: "onnx-pairwise".into(),
            precision: "fp32".into(),
            max_seq_len: 512,
            source: hf(
                "Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                &[("onnx/model.onnx", None), ("tokenizer.json", None)],
            )
            // Pin the upstream revision even before first pull verifies
            // content digests.
            .with_revision("0864ced79bf1ef851bfaf9dd9de0aa54d735d9d0"),
            decision: kredo_decision::DecisionSpec {
                layout: pairwise_nli(&[
                    ("0", "entailment"),
                    ("1", "neutral"),
                    ("2", "contradiction"),
                ]),
            },
            questions: jeff_questions("non-English"),
            router: None,
            provenance: None,
            verification: None,
        },
        Manifest {
            schema: 1,
            name: "nli".into(),
            tag: "latest".into(),
            description: "Zero-shot NLI classifier (DeBERTa-v3-small cross-encoder)".into(),
            engine: "onnx-pairwise".into(),
            precision: "fp32".into(),
            max_seq_len: 512,
            source: hf(
                "Xenova/nli-deberta-v3-small",
                &[("onnx/model.onnx", None), ("tokenizer.json", None)],
            )
            .with_revision("6bc2a55c7c0f7e2bc68de60bb248e523e2612abb"),
            decision: kredo_decision::DecisionSpec {
                layout: pairwise_nli(&[
                    ("0", "contradiction"),
                    ("1", "entailment"),
                    ("2", "neutral"),
                ]),
            },
            questions: vec![],
            router: None,
            provenance: None,
            verification: None,
        },
    ]
}
