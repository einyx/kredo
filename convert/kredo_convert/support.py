"""Real-dataset loader: Tobi-Bueck/customer-support-tickets (61k real support
tickets with queue routing and priority).

Produces JSONL rows for the `support` question set:
    {"text": "<subject>. <body>", "labels": {"queue": "...", "is_urgent": 0/1}}
plus data/support_labels.json (canonical queue label order).
"""

import argparse
import json
import random
from pathlib import Path

import datasets

URGENT_PRIORITIES = {"high", "urgent", "critical"}
MAX_EN_ROWS = 15000  # CPU-friendly cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/support.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-rows", type=int, default=MAX_EN_ROWS)
    args = ap.parse_args()

    ds = datasets.load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    rows = []
    for ex in ds:
        if ex.get("language") != "en":
            continue
        subject = (ex.get("subject") or "").strip()
        body = (ex.get("body") or "").strip().replace("\\n", " ")
        text = f"{subject}. {body}"[:1200]
        queue = (ex.get("queue") or "").strip()
        if not text or not queue:
            continue
        priority = (ex.get("priority") or "").strip().lower()
        rows.append({
            "text": text,
            "labels": {
                "queue": queue,
                "is_urgent": 1.0 if priority in URGENT_PRIORITIES else 0.0,
            },
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.max_rows]

    labels = sorted({r["labels"]["queue"] for r in rows})
    Path("data").mkdir(exist_ok=True)
    Path("data/support_labels.json").write_text(json.dumps(labels, indent=1))

    out = Path(args.out)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    urgent = sum(1 for r in rows if r["labels"]["is_urgent"] > 0)
    print(f"wrote {len(rows)} rows, {len(labels)} queues, {urgent} urgent -> {out}")


if __name__ == "__main__":
    main()
