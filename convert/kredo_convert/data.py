"""Synthetic triage dataset generator.

Produces JSONL rows: {"text": ..., "labels": {question_id: value}} where
choice values are option strings, noul values are 0/1 floats and score
values are floats on the question's scale.

The generator is deliberately template-driven and seeded: the dataset is
reproducible, and parity/eval sets can sample from the same distribution
without leaking (templates are split by base scenario).
"""

import argparse
import json
import random
from pathlib import Path

from kredo_convert.questions import QUESTIONS

SCENARIOS = [
    # (intent, templates about the situation, typical urgency, frustration, refund, churn)
    ("a refund", [
        "I was charged {n} times for my {product} this month and want my money back.",
        "You double-billed me for {product}, please refund the duplicate charge.",
        "I cancelled my {product} but you still charged me. I want a refund.",
        "The {product} charge on my card is wrong, refund it please.",
    ], 0.5, 1.8, 1.0, 0.4),
    ("a bug report", [
        "The {product} app crashes every time I open the {feature}.",
        "Found a bug: {feature} shows stale data after syncing.",
        "{feature} is broken since the last update, it just spins forever.",
        "Export from {feature} produces corrupted files.",
    ], 0.4, 1.4, 0.1, 0.3),
    ("a question about billing", [
        "Can you explain the charge on my invoice for {product}?",
        "How is proration calculated when I upgrade {product}?",
        "Do I get an invoice for the annual {product} plan?",
        "What currency will I be billed in for {product}?",
    ], 0.2, 0.4, 0.1, 0.1),
    ("cancelling the account", [
        "Please cancel my account entirely, I am not using {product}.",
        "I want to close my account and stop all {product} charges.",
        "Cancel my subscription, your {feature} doesn't work for my team.",
        "Shut it all down and delete my data, cancelling {product}.",
    ], 0.3, 1.6, 0.2, 0.9),
    ("praise", [
        "Just wanted to say the new {feature} is amazing, great work!",
        "{product} has been fantastic this quarter, love the {feature}.",
        "Huge thanks to the team, {feature} saved us hours this week.",
        "Best update yet, {product} keeps getting better.",
    ], 0.05, 0.1, 0.0, 0.0),
]

URGENCY_MARKERS = [
    "This is urgent, ", "ASAP: ", "Emergency - ", "I need this resolved today. ",
]
CHURN_MARKERS = [
    " If this isn't fixed I'm switching to a competitor.",
    " Otherwise I will cancel everything.",
    " We're evaluating alternatives right now.",
]
CALM_MARKERS = ["No rush, ", "Whenever you get a chance, ", "Quick one: "]
FRUSTRATED_WORDS = ["unacceptable", "ridiculous", "furious", "fed up", "third time", "still broken"]
PRODUCTS = ["subscription", "workspace plan", "pro plan", "API", "team license"]
FEATURES = ["dashboard", "billing page", "export", "integration", "sync", "report builder"]


def _sample_row(rng: random.Random, scenario) -> dict:
    intent, templates, urgency_base, frustration_base, refund_base, churn_base = scenario
    product = rng.choice(PRODUCTS)
    feature = rng.choice(FEATURES)
    text = rng.choice(templates).format(n=rng.choice(["two", "three", "twice"]), product=product, feature=feature)

    urgent = rng.random() < urgency_base
    if urgent:
        text = rng.choice(URGENCY_MARKERS) + text
        urgency = 1.0
    else:
        urgency = max(0.0, urgency_base - 0.3) if rng.random() < 0.5 else 0.0
        if rng.random() < 0.3:
            text = rng.choice(CALM_MARKERS) + text

    frustration = min(3.0, max(0.0, rng.gauss(frustration_base, 0.5)))
    if frustration > 1.8 and rng.random() < 0.7:
        text += f" This is {rng.choice(FRUSTRATED_WORDS)}."
    if rng.random() < churn_base and intent != "praise":
        text += rng.choice(CHURN_MARKERS)

    churn = 1.0 if rng.choice(CHURN_MARKERS) in text else min(1.0, max(0.0, rng.gauss(churn_base, 0.2)))
    refund = 1.0 if intent == "a refund" else min(1.0, max(0.0, rng.gauss(refund_base, 0.1)))
    if intent == "a refund" and rng.random() < 0.9:
        pass  # text already mentions refund

    return {
        "text": text,
        "labels": {
            "intent": intent,
            "is_urgent": float(urgent),
            "frustration": round(frustration, 3),
            "refund_requested": refund,
            "churn_risk": churn,
        },
    }


def generate(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    per = max(1, n // len(SCENARIOS))
    for scenario in SCENARIOS:
        for _ in range(per):
            rows.append(_sample_row(rng, scenario))
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/triage.jsonl")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(args.n, args.seed)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
