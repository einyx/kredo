"""Log a calibration experiment as a child MLflow run.

Fits per-head temperatures by shelling out to the Rust CLI
(`kredo calibrate --dry-run`) and records the result as a child of a
training run, so the lineage train -> calibrate is queryable in the lab.

The shipped calibration (temperatures written into the manifest plus a
re-recorded verification fixture) remains the Rust `kredo calibrate`
without --dry-run; this lab command only records the experiment.

Usage:
    uv run kredo_convert/calibrate_run.py <model> --eval data/incidents.jsonl \
        [--parent <training_run_id>]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from kredo_convert import lab

LINE = re.compile(
    r"^(?P<qid>\S+)\s+(?P<kind>choice|scalar)\s+T=(?P<t>[\d.]+)\s+"
    r"(?P<metric>nll|mse)=(?P<loss>[\d.]+)\s+n=(?P<n>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--parent", default=None,
                    help="training run id (or id prefix) to attach the child run to")
    ap.add_argument("--kredo", default=os.environ.get("KREDO_BIN", "kredo"),
                    help="kredo binary to invoke (default: $KREDO_BIN or PATH)")
    args = ap.parse_args()

    cmd = [args.kredo, "calibrate", args.model, "--eval", args.eval, "--dry-run"]
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.exit(f"calibration failed:\n{proc.stderr}")

    fits = []
    for line in proc.stdout.splitlines():
        m = LINE.match(line.strip())
        if m:
            fits.append(m.groupdict())
    if not fits:
        sys.exit("could not parse calibrate output — no fits recorded")

    parent = args.parent
    if parent:
        import mlflow

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(f"file://{lab.MLRUNS}")
        parent = lab.resolve(mlflow.tracking.MlflowClient(), parent)

    run = lab.start_run(
        f"calibrate-{args.model}",
        params={
            "model": args.model,
            "eval_set": args.eval,
            "eval_sha256": lab.sha256_file(args.eval),
            "parent_run": args.parent or "",
            **{f"temperature/{f['qid']}": float(f["t"]) for f in fits},
        },
        parent=parent)
    if run:
        lab.log_metrics({f"calib/{f['kind']}_{f['metric']}/{f['qid']}": float(f["loss"])
                         for f in fits})
        import mlflow
        mlflow.end_run()
        print(f"lab: calibration logged ({run.info.run_id[:8]})")
    else:
        print("lab: tracking disabled — fitted temperatures above only")


if __name__ == "__main__":
    main()
