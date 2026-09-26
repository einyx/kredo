//! Golden-fixture tests: real inference must reproduce pinned expectations.
//!
//! These require the referenced model to be pulled; they are `#[ignore]`d by
//! default. Run with `cargo test -p kredo-runner -- --ignored`.

use kredo_api::Question;
use kredo_registry::Registry;
use kredo_runner::Device;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load_fixture(name: &str) -> serde_json::Value {
    let path = fixtures_dir().join(name);
    serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
    )
    .expect("parse fixture")
}

fn run_fixture(file: &str) {
    let fx = load_fixture(file);
    let spec = fx["model"].as_str().expect("fixture model");
    let registry = match std::env::var("KREDO_MODELS") {
        Ok(dir) => Registry::with_root(std::path::PathBuf::from(dir)),
        Err(_) => Registry::open(),
    };
    let local = registry
        .get(spec)
        .expect("model pulled; run `kredo pull` first");
    let engine = kredo_runner::load(&local, &registry, Device::Cpu).expect("load engine");

    let questions: Vec<Question> =
        serde_json::from_value(fx["questions"].clone()).expect("fixture questions");
    let state = fx["input"].as_str().expect("fixture input");
    let answers = engine.decide(state, &questions).expect("decide");

    let expectations = fx["expectations"].as_object().expect("expectations");
    for (id, want) in expectations {
        let answer = answers
            .iter()
            .find(|a| &a.id == id)
            .unwrap_or_else(|| panic!("no answer for {id}"));
        if let Some(top) = want["top"].as_str() {
            let got = answer.top().expect("choice answer");
            assert_eq!(got.label, top, "{spec}/{id}: top label");
        }
        if let Some(min) = want["top_p_gt"].as_f64() {
            let got = answer.top().expect("choice answer");
            assert!(got.p > min, "{spec}/{id}: p={} should exceed {min}", got.p);
        }
    }
}

#[test]
#[ignore = "requires `kredo pull kredo:en` (downloads ~256 MB)"]
fn golden_kredo_en_pairwise() {
    run_fixture("kredo-en-pairwise.json");
}
