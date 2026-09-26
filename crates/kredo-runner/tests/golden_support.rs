//! Golden fixture test for the trained `kredo:support` model (real data).
//!
//! Requires `kredo pull kredo:support` (embedded). Ignored by default.

use kredo_api::{Question, QuestionKind};
use kredo_registry::Registry;
use kredo_runner::Device;

#[test]
#[ignore = "requires `kredo pull kredo:support` (embedded, no download)"]
fn golden_kredo_support() {
    let fx: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/kredo-support.json"),
        )
        .expect("fixture present"),
    )
    .expect("valid fixture");
    let tol = fx["fixture_tolerance"].as_f64().unwrap();

    let registry = match std::env::var("KREDO_MODELS") {
        Ok(dir) => Registry::with_root(std::path::PathBuf::from(dir)),
        Err(_) => Registry::open(),
    };
    let local = registry
        .get("kredo:support")
        .expect("model pulled; run `kredo pull kredo:support` first");
    let engine = kredo_runner::load(&local, &registry, Device::Cpu).expect("load engine");
    let questions: Vec<Question> =
        serde_json::from_value(fx["questions"].clone()).expect("fixture questions");

    for case in fx["cases"].as_array().expect("cases") {
        let answers = engine
            .decide(case["input"].as_str().unwrap(), &questions)
            .expect("decide");
        for want in case["answers"].as_array().expect("answers") {
            let id = want["id"].as_str().unwrap();
            let got = answers.iter().find(|a| a.id == id).unwrap();
            match want["type"].as_str().unwrap() {
                "choice" => {
                    for exp in want["value"].as_array().unwrap() {
                        let p = exp["p"].as_f64().unwrap();
                        let got_p = got
                            .probabilities
                            .iter()
                            .find(|x| x.label == exp["label"].as_str().unwrap())
                            .expect("label present")
                            .p;
                        assert!((got_p - p).abs() <= tol, "{id}");
                    }
                }
                "noul" => {
                    assert_eq!(got.kind, QuestionKind::Noul);
                    assert!((got.p.unwrap() - want["value"].as_f64().unwrap()).abs() <= tol);
                }
                _ => {}
            }
        }
    }
}
