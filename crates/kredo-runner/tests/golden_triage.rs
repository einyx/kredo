//! Golden fixture test for the trained `kredo:triage` model.
//!
//! `fixtures/kredo-triage.json` is emitted by `convert/kredo_convert/parity.py`
//! after the ONNX-vs-PyTorch parity gate; expected values are computed by
//! onnxruntime. The Rust runner must reproduce them within tolerance.
//!
//! Requires `kredo pull kredo:triage` (embedded, no download). Ignored by
//! default so `cargo test` stays hermetic; run with `cargo test -p
//! kredo-runner -- --ignored`.

use kredo_api::{Question, QuestionKind};
use kredo_registry::Registry;
use kredo_runner::Device;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load_questions(v: &serde_json::Value) -> Vec<Question> {
    serde_json::from_value(v.clone()).expect("fixture questions")
}

#[test]
#[ignore = "requires `kredo pull kredo:triage` (embedded, no download)"]
fn golden_kredo_triage() {
    let fx: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("kredo-triage.json")).expect("fixture present"),
    )
    .expect("valid fixture");
    let tol = fx["fixture_tolerance"].as_f64().unwrap();

    let registry = match std::env::var("KREDO_MODELS") {
        Ok(dir) => Registry::with_root(std::path::PathBuf::from(dir)),
        Err(_) => Registry::open(),
    };
    let local = registry
        .get("kredo:triage")
        .expect("model pulled; run `kredo pull kredo:triage` first");
    let engine = kredo_runner::load(&local, &registry, Device::Cpu).expect("load engine");
    let questions: Vec<Question> = load_questions(&fx["questions"]);

    for case in fx["cases"].as_array().expect("cases") {
        let state = case["input"].as_str().unwrap();
        let answers = engine.decide(state, &questions).expect("decide succeeds");
        for want in case["answers"].as_array().expect("answers") {
            let id = want["id"].as_str().unwrap();
            let got = answers
                .iter()
                .find(|a| a.id == id)
                .unwrap_or_else(|| panic!("no answer for {id}"));
            match want["type"].as_str().unwrap() {
                "choice" => {
                    let expected = want["value"].as_array().unwrap();
                    for exp in expected {
                        let label = exp["label"].as_str().unwrap();
                        let p = exp["p"].as_f64().unwrap();
                        let got_p = got
                            .probabilities
                            .iter()
                            .find(|x| x.label == label)
                            .unwrap_or_else(|| panic!("{id}/{label} missing"))
                            .p;
                        assert!(
                            (got_p - p).abs() <= tol,
                            "{id}/{label}: got {got_p}, want {p} (state: {state})"
                        );
                    }
                }
                "score" => {
                    let got_score = got.score.as_ref().expect("score answer").value;
                    let p = want["value"]["value"].as_f64().unwrap();
                    assert!(
                        (got_score - p).abs() <= tol,
                        "{id}: got {got_score}, want {p} (state: {state})"
                    );
                }
                "noul" => {
                    let kind = QuestionKind::Noul;
                    assert_eq!(got.kind, kind);
                    let got_p = got.p.expect("noul answer");
                    let p = want["value"].as_f64().unwrap();
                    assert!(
                        (got_p - p).abs() <= tol,
                        "{id}: got {got_p}, want {p} (state: {state})"
                    );
                }
                other => panic!("unknown kind {other}"),
            }
        }
    }
}
