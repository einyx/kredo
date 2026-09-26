# convert/

Build-time Python pipeline for kredo decision models: dataset generation,
multi-head training, ONNX export, the parity gate, and packaging.

## Pipeline

```sh
cd convert
uv sync
uv run kredo_convert/data.py --out data/triage.jsonl --n 6000
uv run kredo_convert/train.py --data data/triage.jsonl --out out/triage
uv run kredo_convert/export.py --model out/triage --out out/triage.onnx
uv run kredo_convert/parity.py --model out/triage --onnx out/triage.onnx
uv run kredo_convert/package.py --model out/triage --onnx out/triage.onnx
```

## Contract

- The ONNX graph takes `input_ids` / `attention_mask` (dynamic batch/seq)
  and emits one output tensor per question head, named `head_<question_id>`
  with shape `[batch, n]` (`n`=options for `choice`, 2 for `noul`, 1 for
  `score`).
- `parity.py` is a hard gate: every head must match PyTorch within 1e-4 on
  the eval distribution before packaging; it also emits the golden fixture
  the Rust-side test (`cargo test -p kredo-runner -- --ignored`) checks.
- `package.py` writes artifacts into `registry/v1/` and prints digests; the
  Rust embedded library (`crates/kredo-registry/src/embedded.rs`) serves them
  so `kredo pull kredo:triage` works offline and byte-identically in CI.

## Adding a model

1. Define the question set in `kredo_convert/questions.py`.
2. Extend the dataset generator (or point `--data` at your own JSONL).
3. Train, export, pass parity, package; wire the manifest into the Rust
   library with the printed digests.
