//! Debug: print raw logits for one (premise, hypothesis) pair.

use kredo_registry::Registry;

fn main() -> anyhow::Result<()> {
    let registry = Registry::with_root(std::path::PathBuf::from(
        std::env::var("KREDO_MODELS").unwrap(),
    ));
    let model = registry.get("kredo:en")?;
    let engine = kredo_runner::load(&model, &registry, kredo_runner::Device::Cpu)?;
    let qs = vec![kredo_api::Question {
        id: "refund".into(),
        kind: kredo_api::QuestionKind::Noul,
        prompt: "The user explicitly asks for a refund".into(),
        options: vec![],
        min: None,
        max: None,
    }];
    let answers = engine.decide("I was charged twice this month and want a refund.", &qs)?;
    println!("{answers:?}");
    Ok(())
}
