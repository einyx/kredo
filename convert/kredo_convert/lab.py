"""The kredo model lab: MLflow-backed experiment tracking and comparison.

Division of labor: the lab is the *inner loop* — it helps you choose a
model. Shipping proof lives in the manifest (provenance, verification
fixtures) and in `kredo verify` / `kredo shadow`. MLflow here is a local,
file-backed tracking store (convert/mlruns/) — no server, no registry.

Everything is failure-tolerant: when tracking is unavailable, training and
calibration behave exactly as without the lab (a warning is printed).

CLI:
    uv run kredo_convert/lab.py list [--limit 20]
    uv run kredo_convert/lab.py compare <run_id_a> <run_id_b>
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

MLRUNS = Path(__file__).resolve().parent.parent / "mlruns"


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _available() -> bool:
    if not _importable():
        return False
    # MLflow 3.x gates the file backend behind an explicit opt-in; the lab
    # deliberately uses a plain directory (see plan: file store, no server).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return True


def _importable() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def start_run(name: str, params: dict | None = None, parent: str | None = None):
    """Open a tolerant MLflow run context manager; None when tracking off."""
    if not _available():
        print("lab: mlflow not installed — tracking disabled", file=sys.stderr)
        return None
    import mlflow

    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLRUNS}")
    mlflow.set_experiment("kredo")
    tags = {"kredo.pipeline": "convert"}
    try:
        run = mlflow.start_run(run_name=name, tags=tags, parent_run_id=parent)
        if params:
            mlflow.log_params(params)
        return run
    except Exception as e:  # noqa: BLE001 — the lab must never block training
        print(f"lab: tracking unavailable ({e}) — continuing without", file=sys.stderr)
        return None


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if not _available():
        return
    import mlflow

    active = mlflow.active_run()
    if not active:
        return
    try:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
    except Exception as e:  # noqa: BLE001
        print(f"lab: metric logging failed ({e})", file=sys.stderr)


def log_artifacts(path: str | os.PathLike) -> None:
    if not _available():
        return
    import mlflow

    active = mlflow.active_run()
    if not active:
        return
    try:
        mlflow.log_artifacts(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"lab: artifact logging failed ({e})", file=sys.stderr)


def _short_params(p: dict) -> str:
    keep = ["base", "lr", "epochs", "batch_size", "seed", "dataset",
            "temperature", "parent_run"]
    return " ".join(f"{k}={v}" for k, v in p.items() if k in keep and v)


def list_runs(limit: int = 20) -> None:
    if not _available():
        sys.exit("lab: mlflow is not installed")
    import mlflow

    mlflow.set_tracking_uri(f"file://{MLRUNS}")
    exp = mlflow.get_experiment_by_name("kredo")
    if exp is None:
        sys.exit("lab: no runs yet")
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"],
                              max_results=limit)
    if runs.empty:
        sys.exit("lab: no runs yet")
    for _, r in runs.iterrows():
        name = r.get("tags.mlflow.runName", "?")
        status = r.get("status", "?")
        print(f"{r.run_id[:8]}  {status:<9}{name}  {_short_params(r.to_dict())}")


def compare(run_a: str, run_b: str) -> None:
    if not _available():
        sys.exit("lab: mlflow is not installed")
    import mlflow

    mlflow.set_tracking_uri(f"file://{MLRUNS}")
    client = mlflow.tracking.MlflowClient()

    def fetch(prefix: str) -> tuple[dict, dict, str]:
        rid = resolve(client, prefix)
        run = client.get_run(rid)
        return dict(run.data.params), dict(run.data.metrics), rid

    pa, ma, rid_a = fetch(run_a)
    pb, mb, rid_b = fetch(run_b)
    print(f"a = {rid_a}  {_short_params(pa)}")
    print(f"b = {rid_b}  {_short_params(pb)}")
    print()
    keys = sorted(set(ma) | set(mb))
    print(f"{'metric':<24}{'a':>12}{'b':>12}{'delta':>12}")
    regressions = []
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        if va is None or vb is None:
            print(f"{k:<24}{_fmt(va):>12}{_fmt(vb):>12}{'—':>12}")
            continue
        d = vb - va
        # Higher-is-better metrics (accuracy/noul acc); mae/loss are errors.
        higher_better = not any(s in k for s in ("mae", "loss"))
        flag = ""
        if abs(d) > 1e-9:
            good = d > 0 if higher_better else d < 0
            flag = " ✓" if good else " ✗ REGRESSION"
            if not good:
                regressions.append(k)
        print(f"{k:<24}{va:>12.4f}{vb:>12.4f}{d:>+12.4f}{flag}")
    if regressions:
        print(f"\nb regresses on: {', '.join(regressions)}")
        sys.exit(1)
    print("\nb does not regress on any logged metric")


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def resolve(client, prefix: str) -> str:
    runs = client.search_runs(
        [client.get_experiment_by_name("kredo").experiment_id],
        filter_string=f"run_id LIKE '{prefix}%'", max_results=2)
    if len(runs) != 1:
        sys.exit(f"lab: run prefix `{prefix}` matches {len(runs)} runs")
    return runs[0].info.run_id


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="kredo model lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--limit", type=int, default=20)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("run_a")
    cmp_.add_argument("run_b")
    args = ap.parse_args()
    if args.cmd == "list":
        list_runs(args.limit)
    else:
        compare(args.run_a, args.run_b)
