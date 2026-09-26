"""Parity gate: ONNX must reproduce PyTorch on the eval distribution.

Runs N states through both the torch model and onnxruntime, asserts every
head matches within tolerance, then writes:
  - the golden fixture for the Rust test-suite (fixtures/), and
  - the full verification document (provenance + fixture) that ships with
    the model and backs `kredo verify` / --require-verified.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from kredo_convert.data import generate
from kredo_convert.questions import get, head_output
from kredo_convert.train import MultiHeadModel


def torch_heads(model, tok, texts, questions, max_len=128):
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        return model(enc["input_ids"], enc["attention_mask"])


def onnx_heads(sess, tok, texts, questions, max_len=128):
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_len)
    ids = np.array(enc["input_ids"], dtype=np.int64)
    mask = np.array(enc["attention_mask"], dtype=np.int64)
    names = [head_output(q[0]) for q in questions]
    outs = sess.run(names, {"input_ids": ids, "attention_mask": mask})
    return dict(zip(names, outs))


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


import math  # noqa: E402


def postprocess(kind, raw, options, mn, mx):
    if kind == "choice":
        ps = softmax(raw)
        return [
            {"label": options[i], "p": round(p, 6)}
            for i, p in sorted(enumerate(ps), key=lambda kv: -kv[1])
        ]
    if kind == "score":
        unit = 1 / (1 + math.exp(-raw[0]))
        return {"value": round(mn + (mx - mn) * unit, 6), "normalized": round(unit, 6), "min": mn, "max": mx}
    if len(raw) >= 2:
        return round(softmax(raw)[1], 6)
    return round(1 / (1 + math.exp(-raw[0])), 6)


def collect_metrics(torch_out, rows, questions) -> dict:
    gold_map = {q[0]: q for q in questions}
    metrics: dict[str, float] = {}
    for qid, kind, _, options, mn, mx in questions:
        pred = torch_out[qid].numpy()
        if kind == "choice":
            gold = np.array([options.index(r["labels"][qid]) for r in rows])
            metrics[f"{qid}.accuracy"] = round(float((pred.argmax(-1) == gold).mean()), 4)
        elif kind == "noul":
            gold = np.array([int(r["labels"][qid] > 0.5) for r in rows])
            metrics[f"{qid}.accuracy"] = round(float((pred.argmax(-1) == gold).mean()), 4)
        else:
            gold = np.array([float(r["labels"][qid]) for r in rows])
            predv = 1 / (1 + np.exp(-pred.squeeze(-1))) * (mx - mn) + mn
            metrics[f"{qid}.mae"] = round(float(np.abs(predv - gold).mean()), 4)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="out/triage")
    ap.add_argument("--onnx", default="out/triage.onnx")
    ap.add_argument("--set", default="triage")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--fixture-out", default=None)
    ap.add_argument("--verification-out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--pipeline", default="convert v0.1.0")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    questions = get(args.set)
    fixture_out = Path(args.fixture_out or f"fixtures/kredo-{args.set}.json")
    verification_out = Path(args.verification_out or f"out/verification-{args.set}.json")

    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text())
    model = MultiHeadModel(cfg["base"], questions)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    # Eval distribution: synthetic for triage, real rows for banking.
    if args.set == "triage":
        rows = generate(args.n, seed=args.seed)
    else:
        path = Path(f"data/{args.set}.jsonl")
        all_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = all_rows[-args.n:]  # hold-out slice
    texts = [r["text"] for r in rows]

    torch_out = torch_heads(model, tok, texts, questions)
    onnx_out = onnx_heads(sess, tok, texts, questions)

    worst = 0.0
    for qid, _, _, _, _, _ in questions:
        name = head_output(qid)
        diff = float(np.abs(torch_out[qid].numpy() - onnx_out[name]).max())
        worst = max(worst, diff)
        assert diff <= args.tol, f"parity failed for {qid}: max diff {diff} > {args.tol}"
    print(f"parity OK: worst head diff {worst:.2e} over {len(texts)} states (tol {args.tol})")

    metrics = collect_metrics(torch_out, rows, questions)
    print("metrics:", metrics)

    # Golden fixture cases straight from onnxruntime.
    if args.set == "triage":
        fixture_states = [r["text"] for r in generate(3, seed=1234)]
    else:
        fixture_states = [r["text"] for r in rows[:3]]
    onnx_fix = onnx_heads(sess, tok, fixture_states, questions)
    cases = []
    for si, text in enumerate(fixture_states):
        answers = []
        for qid, kind, prompt, options, mn, mx in questions:
            raw = onnx_fix[head_output(qid)][si]
            answers.append({
                "id": qid,
                "type": kind,
                "value": postprocess(kind, [float(x) for x in raw], options, mn, mx),
            })
        cases.append({"input": text, "answers": answers})

    fixture = {
        "model": f"kredo:{args.set}",
        "note": "expected values computed by onnxruntime at export time; "
        "the Rust runner must match within fixture_tolerance",
        "fixture_tolerance": args.tol,
        "questions": [
            {"id": qid, "type": kind, "prompt": prompt, "options": options or [], "min": mn, "max": mx}
            for qid, kind, prompt, options, mn, mx in questions
        ],
        "cases": cases,
    }
    fixture_out.parent.mkdir(parents=True, exist_ok=True)
    fixture_out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"golden fixture -> {fixture_out}")

    # Verification document shipped with the model.
    raw = fixture_out.read_bytes()
    verification = {
        "provenance": {
            "dataset": args.dataset or f"{args.set} eval distribution (seed {args.seed})",
            "seed": args.seed,
            "pipeline": args.pipeline,
            "metrics": metrics,
        },
        "verification": {
            "fixture_digest": hashlib.sha256(raw).hexdigest(),
            "tolerance": args.tol,
            "verified_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "cases": len(cases),
        },
        "questions": fixture["questions"],
        "cases": cases,
    }
    verification_out.write_text(json.dumps(verification, indent=2) + "\n")
    print(f"verification doc -> {verification_out} "
          f"(digest {verification['verification']['fixture_digest'][:16]}...)")


if __name__ == "__main__":
    main()
