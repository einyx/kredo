"""Train a multi-head decision model.

Architecture: any HF encoder (default: prajjwal1/bert-tiny) + a pooled CLS
head feeding one linear head per question (choice: n logits, noul: 2 logits,
score: 1 logit squashed through sigmoid onto [min, max]).

The training loss is the sum of per-head losses; score heads regress the
sigmoid output against the normalized target.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from kredo_convert.questions import get


class MultiHeadModel(nn.Module):
    def __init__(self, encoder_name: str, questions):
        super().__init__()
        self.questions = questions
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict()
        for qid, kind, _, options, _, _ in questions:
            n = len(options) if kind == "choice" else (2 if kind == "noul" else 1)
            self.heads[qid] = nn.Linear(hidden, n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return {qid: head(cls) for qid, head in self.heads.items()}


class TriageDataset(Dataset):
    def __init__(self, rows, tokenizer, questions, max_len: int = 128):
        self.rows = rows
        self.tok = tokenizer
        self.questions = questions
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        enc = self.tok(row["text"], truncation=True, max_length=self.max_len, padding="max_length")
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }
        for qid, kind, _, options, mn, mx in self.questions:
            v = row["labels"][qid]
            if kind == "choice":
                item[qid] = torch.tensor(options.index(v))
            elif kind == "noul":
                item[qid] = torch.tensor(float(v))
            else:
                item[qid] = torch.tensor(float((v - mn) / (mx - mn)))
        return item


def loss_fn(outputs, batch, questions):
    loss = 0.0
    for qid, kind, _, _, _, _ in questions:
        target = batch[qid]
        if kind == "choice":
            loss = loss + nn.functional.cross_entropy(outputs[qid], target)
        elif kind == "noul":
            loss = loss + nn.functional.cross_entropy(outputs[qid], target.long())
        else:
            pred = torch.sigmoid(outputs[qid].squeeze(-1))
            loss = loss + nn.functional.mse_loss(pred, target)
    return loss


@torch.no_grad()
def evaluate(model, loader, questions) -> dict:
    model.eval()
    stats = {}
    for batch in loader:
        outputs = model(batch["input_ids"], batch["attention_mask"])
        for qid, kind, _, options, mn, mx in questions:
            if kind == "choice":
                pred = outputs[qid].argmax(-1)
                gold = batch[qid]
                acc = (pred == gold).float().mean().item()
                s = stats.setdefault(qid, [0.0, 0])
                s[0] += acc * len(gold)
                s[1] += len(gold)
            elif kind == "noul":
                pred = outputs[qid].argmax(-1)
                s = stats.setdefault(qid, [0.0, 0])
                s[0] += (pred == batch[qid].long()).float().sum().item()
                s[1] += len(batch[qid])
            else:
                pred = torch.sigmoid(outputs[qid].squeeze(-1)) * (mx - mn) + mn
                err = (pred - batch[qid]).abs().mean().item()
                s = stats.setdefault(qid, [0.0, 0])
                s[0] += err * len(batch[qid])
                s[1] += len(batch[qid])
    report = {}
    for qid, kind, _, _, _, _ in questions:
        total, n = stats[qid]
        # Match the manifest provenance metric naming: acc/<qid> for
        # classification heads, mae/<qid> for score heads.
        key = f"mae/{qid}" if kind == "score" else f"acc/{qid}"
        report[key] = round(total / max(n, 1), 4)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/triage.jsonl")
    ap.add_argument("--base", default="google/bert_uncased_L-2_H-128_A-2")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--set", default="triage")
    ap.add_argument("--out", default="out/triage")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=11,
                    help="train/val split seed (the dataset generator seed is separate)")
    ap.add_argument("--no-lab", action="store_true",
                    help="skip MLflow tracking entirely")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val_frac)
    val_rows = [rows[i] for i in idx[:n_val]]
    train_rows = [rows[i] for i in idx[n_val:]]

    # Lab: one parent run per training. Failure-tolerant — see lab.py.
    run = None
    if not args.no_lab:
        from kredo_convert import lab
        run = lab.start_run(
            f"train-{args.set}",
            params={
                "set": args.set, "base": args.base, "tokenizer": args.tokenizer,
                "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
                "seed": args.seed, "val_frac": args.val_frac,
                "dataset": args.data,
                "dataset_sha256": lab.sha256_file(args.data),
                "rows": len(rows), "train_rows": len(train_rows),
                "val_rows": len(val_rows),
            })

    questions = get(args.set)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train_ds = TriageDataset(train_rows, tok, questions)
    val_ds = TriageDataset(val_rows, tok, questions)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = MultiHeadModel(args.base, questions)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader):
            opt.zero_grad()
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(outputs, batch, questions)
            loss.backward()
            opt.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        print(f"epoch {epoch} mean loss {total / max(1, len(train_loader)):.4f}")
        report = evaluate(model, val_loader, questions)
        print(f"epoch {epoch} eval {report}")
        if run:
            from kredo_convert import lab
            lab.log_metrics({"train_loss": total / max(1, len(train_loader))}, step=epoch)
            lab.log_metrics(report, step=epoch)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tok.save_pretrained(out)
    (out / "config.json").write_text(json.dumps({"base": args.base, "tokenizer": args.tokenizer, "qset": args.set, "questions_meta": [
        {"id": q[0], "kind": q[1], "options": q[3], "min": q[4], "max": q[5]} for q in questions
    ]}, indent=2))
    print(f"saved -> {out}")
    if run:
        from kredo_convert import lab
        lab.log_artifacts(out)
        import mlflow  # tolerated import; lab already checked availability
        mlflow.end_run()
        print(f"lab: run logged ({run.info.run_id[:8]}) — compare with `lab.py compare`")


if __name__ == "__main__":
    main()
