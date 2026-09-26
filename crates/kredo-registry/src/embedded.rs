/// Embedded model artifacts.
///
/// Small trained graphs ship inside the binary so `kredo pull` works offline
/// and CI exercises byte-identical models. Large third-party checkpoints
/// still stream from Hugging Face with pinned revisions/digests.
pub fn blob(name: &str, tag: &str, path: &str) -> Option<&'static [u8]> {
    match (name, tag, path) {
        ("kredo", "triage", "model.onnx") => Some(include_bytes!(
            "../../../registry/v1/kredo/triage/model.onnx"
        )),
        ("kredo", "triage", "tokenizer.json") => Some(include_bytes!(
            "../../../registry/v1/kredo/triage/tokenizer.json"
        )),
        ("kredo", "triage", "verification.json") => Some(include_bytes!(
            "../../../registry/v1/kredo/triage/verification.json"
        )),
        ("kredo", "support", "model.onnx") => Some(include_bytes!(
            "../../../registry/v1/kredo/support/model.onnx"
        )),
        ("kredo", "support", "tokenizer.json") => Some(include_bytes!(
            "../../../registry/v1/kredo/support/tokenizer.json"
        )),
        ("kredo", "support", "verification.json") => Some(include_bytes!(
            "../../../registry/v1/kredo/support/verification.json"
        )),
        _ => None,
    }
}
