"""Question sets.

`get(name)` returns the question list for a trained model family. The
banking set's 77 labels are produced by the dataset loader (data.py
`banking`) into data/banking_labels.json, so the model, manifest and
verification document always share one label order.
"""

import json
from pathlib import Path

# (id, kind, prompt, options, min, max)
QUESTIONS_TRIAGE = [
    (
        "intent",
        "choice",
        "What does the user want?",
        ["a refund", "a bug report", "a question about billing", "cancelling the account", "praise"],
        None,
        None,
    ),
    ("is_urgent", "noul", "This message is urgent and needs a same-day response", None, None, None),
    ("frustration", "score", "How frustrated is the user?", ["not frustrated at all", "extremely frustrated"], 0.0, 3.0),
    ("refund_requested", "noul", "The user explicitly asks for a refund", None, None, None),
    ("churn_risk", "noul", "The user is at risk of cancelling their subscription", None, None, None),
]

LABELS_FILE = Path("data/support_labels.json")


def support_labels() -> list[str]:
    if not LABELS_FILE.exists():
        raise SystemExit("run `uv run python -m kredo_convert.support` first")
    return json.loads(LABELS_FILE.read_text())


def questions_support() -> list:
    return [
        (
            "queue",
            "choice",
            "Which support queue should handle this ticket?",
            support_labels(),
            None,
            None,
        ),
        (
            "is_urgent",
            "noul",
            "This ticket is high priority and needs an immediate response",
            None,
            None,
            None,
        ),
    ]


def get(name: str):
    if name == "triage":
        return QUESTIONS_TRIAGE
    if name == "support":
        return questions_support()
    raise SystemExit(f"unknown question set {name!r} (triage|support)")


def head_output(qid: str) -> str:
    return f"head_{qid}"


def head_labels(qid: str, questions):
    for q in questions:
        if q[0] == qid:
            _, kind, _, options, mn, mx = q
            if kind == "choice":
                return list(options)
            if kind == "score":
                return [f"min={mn}", f"max={mx}"]
            return ["yes", "no"]
    raise KeyError(qid)


def heads(questions):
    return [
        {"question": q[0], "output": head_output(q[0]), "labels": head_labels(q[0], questions)}
        for q in questions
    ]


# Back-compat for the triage pipeline.
QUESTIONS = QUESTIONS_TRIAGE
HEADS = heads(QUESTIONS_TRIAGE)
