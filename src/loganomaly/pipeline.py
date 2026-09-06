"""End-to-end anomaly detection pipeline.

Flow: load logs -> embed -> index into Qdrant -> run all detectors ->
evaluate against ground truth -> cluster flagged anomalies for triage ->
write reports.

Outputs:
  results/detector_benchmark.md — metric table across all detectors
  results/scores.csv            — per-record scores from every detector
  results/triage_report.md      — top alerts with nearest neighbours and
                                  anomaly clusters (what an on-call person
                                  would actually read)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loganomaly.config import settings
from loganomaly.detection.detectors import (
    DETECTOR_REGISTRY,
    cluster_embeddings,
    normalize_scores,
    summarize_clusters,
)
from loganomaly.evaluation.metrics import evaluate_detector, recall_at_k_per_type
from loganomaly.models import LogRecord, load_jsonl
from loganomaly.storage.vector_store import VectorStore


def build_store() -> VectorStore:
    return VectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


class AnomalyPipeline:
    def __init__(self, store: VectorStore):
        # Embedder is imported lazily so tooling/tests that only touch the
        # pipeline module never load torch.
        from loganomaly.embedding.embedder import Embedder

        self.store = store
        self.embedder = Embedder(settings.embedding_model)
        self.records: list[LogRecord] = []
        self.vectors: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def ingest(self) -> int:
        """Load, embed and index the log file."""
        self.records = load_jsonl(settings.logs_file)
        self.vectors = self.embedder.embed([r.text for r in self.records])
        self.store.recreate_collection(settings.collection, self.embedder.dimension)
        self.store.upsert_records(settings.collection, self.records, self.vectors)
        return len(self.records)

    # ------------------------------------------------------------------ #
    def detect(self) -> pd.DataFrame:
        """Run every detector; return a DataFrame with one score column each
        (normalised to [0, 1], higher = more anomalous)."""
        assert self.vectors is not None, "call ingest() first"
        frame = pd.DataFrame(
            {
                "record_id": [r.record_id for r in self.records],
                "text": [r.text for r in self.records],
                "is_anomaly": [r.is_anomaly for r in self.records],
                "anomaly_type": [r.anomaly_type for r in self.records],
            }
        )
        # sklearn-based detectors operate on the local embedding matrix.
        for name, detector_cls in DETECTOR_REGISTRY.items():
            raw = detector_cls().fit_score(self.vectors)
            frame[name] = normalize_scores(raw)
        # The k-NN distance detector is served by Qdrant itself.
        raw_knn = self.store.knn_mean_distance(settings.collection, self.vectors, k=10)
        frame["knn_qdrant"] = normalize_scores(raw_knn)
        return frame

    # ------------------------------------------------------------------ #
    def evaluate(self, frame: pd.DataFrame, k: int = 50) -> pd.DataFrame:
        """Benchmark table: one row per detector.

        `k` defaults to roughly the number of injected anomalies so
        precision@k reflects a realistic review budget.
        """
        labels = frame["is_anomaly"].to_numpy(dtype=int)
        rows = []
        for name in self._detector_columns(frame):
            scores = frame[name].to_numpy()
            row = {"detector": name, **evaluate_detector(labels, scores, k)}
            row.update(
                {
                    f"recall@{k}:{t}": v
                    for t, v in recall_at_k_per_type(
                        labels, frame["anomaly_type"].tolist(), scores, k
                    ).items()
                }
            )
            rows.append(row)
        return pd.DataFrame(rows).round(3)

    # ------------------------------------------------------------------ #
    def write_reports(self, frame: pd.DataFrame, benchmark: pd.DataFrame, top_n: int = 15) -> None:
        """Persist the benchmark table, raw scores and a triage report."""
        settings.results_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.results_dir / "scores.csv", index=False)
        (settings.results_dir / "detector_benchmark.md").write_text(
            benchmark.to_markdown(index=False), encoding="utf-8"
        )
        (settings.results_dir / "triage_report.md").write_text(
            self._triage_report(frame, top_n), encoding="utf-8"
        )

    def _triage_report(self, frame: pd.DataFrame, top_n: int) -> str:
        """Human-readable report: top alerts (by best PR-AUC detector proxy:
        knn_qdrant) with nearest neighbours, plus anomaly clusters."""
        assert self.vectors is not None
        lines = ["# Triage Report", ""]

        # --- Top alerts with nearest-neighbour context -------------------- #
        lines += [f"## Top {top_n} alerts (knn_qdrant score)", ""]
        top = frame.sort_values("knn_qdrant", ascending=False).head(top_n)
        for _, row in top.iterrows():
            record_index = frame.index[frame["record_id"] == row["record_id"]][0]
            lines.append(f"### `{row['record_id']}`  score={row['knn_qdrant']:.3f}")
            lines.append(f"> {row['text']}")
            lines.append("")
            lines.append("Nearest neighbours (for context — is this a known pattern?):")
            for nb in self.store.neighbours(
                settings.collection, self.vectors[record_index], limit=4
            )[1:]:  # skip self
                lines.append(f"- ({nb['score']:.3f}) {nb['level']} {nb['service']}: {nb['message'][:110]}")
            lines.append("")

        # --- Cluster view over flagged anomalies --------------------------- #
        flagged_idx = np.argsort(frame["knn_qdrant"].to_numpy())[::-1][: top_n * 3]
        labels = cluster_embeddings(self.vectors[flagged_idx])
        summaries = summarize_clusters(
            labels, self.vectors[flagged_idx], [frame["text"].iloc[i] for i in flagged_idx]
        )
        lines += ["## Alert clusters (DBSCAN over the top alerts)", ""]
        lines.append("| cluster | size | representative log line |")
        lines.append("|---|---|---|")
        for s in summaries:
            rep = str(s["representative"]).replace("|", "\\|")[:130]
            lines.append(f"| {s['cluster']} | {s['size']} | {rep} |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _detector_columns(frame: pd.DataFrame) -> list[str]:
        reserved = {"record_id", "text", "is_anomaly", "anomaly_type"}
        return [c for c in frame.columns if c not in reserved]
