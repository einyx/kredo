"""Package a converted model: digests, registry artifacts, manifest snippet.

Writes the ONNX graph and tokenizer into `registry/v1/kredo/<tag>/` inside the
repository (committed and served by the embedded-library path in
crates/kredo-registry) and prints the Rust-side FileSpec digests.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from kredo_convert.questions import HEADS, QUESTIONS


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="out/triage")
    ap.add_argument("--onnx", default="out/triage.onnx")
    ap.add_argument("--name", default="kredo")
    ap.add_argument("--tag", default="triage")
    ap.add_argument("--set", default="triage")
    ap.add_argument("--registry-dir", default="registry/v1")
    args = ap.parse_args()

    model_dir = Path(args.model)
    onnx = Path(args.onnx)
    dest = Path(args.registry_dir) / args.name / args.tag
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy(onnx, dest / "model.onnx")
    tok_file = model_dir / "tokenizer.json"
    if tok_file.exists():
        shutil.copy(tok_file, dest / "tokenizer.json")

    print("# files")
    for f in sorted(dest.iterdir()):
        print(f'  ("{f.name}", "{sha256_file(f)}"),')

    cfg = json.loads((model_dir / "config.json").read_text())
    questions = get(args.set)
    hs = heads(questions)
    print("\n# Rust manifest fragment")
    print(json.dumps({
        "engine": "onnx-multihead",
        "max_seq_len": 128,
        "decision": {"layout": {"kind": "multihead", "heads": hs}},
        "questions": [
            {"id": q[0], "type": q[1], "prompt": q[2], "options": q[3] or [],
             "min": q[4], "max": q[5]}
            for q in questions
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
