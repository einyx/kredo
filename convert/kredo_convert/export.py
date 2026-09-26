"""Export a trained multi-head model to ONNX.

Outputs one tensor per head named `head_<question_id>` (shape [1, n]) plus
the standard `input_ids` / `attention_mask` inputs, matching the Rust
runner's native multi-head path.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from kredo_convert.questions import get, head_output
from kredo_convert.train import MultiHeadModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="out/triage")
    ap.add_argument("--out", default="out/triage.onnx")
    args = ap.parse_args()

    model_dir = Path(args.model)
    cfg = __import__("json").loads((model_dir / "config.json").read_text())
    questions = get(cfg["qset"])
    model = MultiHeadModel(cfg["base"], questions)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()

    tok = AutoTokenizer.from_pretrained(model_dir)
    sample = tok("parity check input", return_tensors="pt", padding="max_length", max_length=16)

    output_names = [head_output(q[0]) for q in questions]
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            args.out,
            input_names=["input_ids", "attention_mask"],
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                **{n: {0: "batch"} for n in output_names},
            },
            opset_version=17,
        )
    out_path = Path(args.out)
    print(f"exported -> {out_path}")

    # The exporter may emit weights as an external .data file; inline them so
    # the graph is a single self-contained artifact.
    data_file = out_path.with_name(out_path.name + ".data")
    if data_file.exists():
        import onnx
        from onnx.external_data_helper import convert_model_from_external_data

        m = onnx.load(str(out_path), load_external_data=True)
        for tensor in m.graph.initializer:
            tensor.ClearField("data_location")
        onnx.save(m, str(out_path))
        data_file.unlink()
        print(f"inlined external weights -> {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
