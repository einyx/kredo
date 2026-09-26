"""kredo model conversion: dataset, training, ONNX export, parity, packaging.

Each module is runnable directly:

    uv run kredo_convert/data.py --out data/triage.jsonl
    uv run kredo_convert/train.py --data data/triage.jsonl --out out/triage
    uv run kredo_convert/export.py --model out/triage --out out/triage.onnx
    uv run kredo_convert/parity.py --model out/triage --onnx out/triage.onnx
    uv run kredo_convert/package.py --model out/triage --onnx out/triage.onnx
"""

from .questions import QUESTIONS, HEADS

__all__ = ["QUESTIONS", "HEADS"]
