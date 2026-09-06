"""Typed data models shared across the pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LogRecord:
    """One log line, plus the ground-truth label used only for evaluation.

    In production the label obviously does not exist — the detectors never
    see it. It is carried along solely so the benchmark can compute
    ROC-AUC / precision@k afterwards.
    """

    record_id: str       # stable id, e.g. "log-000042"
    timestamp: str       # ISO-8601
    level: str           # INFO | WARN | ERROR | ...
    service: str         # emitting service name
    message: str         # the raw log text that gets embedded
    is_anomaly: bool     # ground truth (evaluation only!)
    anomaly_type: str = ""  # which injector produced it (evaluation only)

    @property
    def text(self) -> str:
        """The string that is embedded. Level and service are prepended
        because they carry signal (e.g. an unknown service name is itself
        suspicious) and keep the embedding self-contained."""
        return f"{self.level} {self.service}: {self.message}"


def save_jsonl(records: list[LogRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record)) + "\n")


def load_jsonl(path: Path) -> list[LogRecord]:
    records = [LogRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No log records found in {path}")
    return records
