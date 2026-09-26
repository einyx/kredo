use kredo_registry::Registry;
fn main() {
    let reg = Registry::with_root(std::path::PathBuf::from(
        std::env::var("KREDO_MODELS").unwrap(),
    ));
    let m = reg.get("kredo:triage").unwrap();
    let e = m
        .digests
        .iter()
        .find(|(p, _)| p.ends_with("tokenizer.json"))
        .unwrap();
    let tok = tokenizers::Tokenizer::from_file(reg.blob_path(e.1)).unwrap();
    let mut tok = tok;
    tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length: 128,
        ..Default::default()
    }))
    .unwrap();
    tok.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(128),
        ..Default::default()
    }));
    let enc = tok.encode("hello world", true).unwrap();
    println!(
        "len={} ids[..5]={:?}",
        enc.get_ids().len(),
        &enc.get_ids()[..5.min(enc.get_ids().len())]
    );
    println!("pad_token={:?}", tok.get_padding());
}
