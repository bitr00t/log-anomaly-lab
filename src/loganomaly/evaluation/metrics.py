"""Evaluation metrics for anomaly detectors.

Anomaly detection is heavily imbalanced (~3% positives here), which drives
the metric choice:

  * ROC-AUC          — threshold-free ranking quality; fine for comparison
                       but optimistic under class imbalance.
  * PR-AUC (average  — the honest headline metric for rare positives: it
    precision)         only looks at how well anomalies rise to the top.
  * precision@k      — the operational metric: "if an engineer reviews the
                       top k alerts, how many are real?"
  * recall per       — which anomaly *types* a detector misses; a detector
    anomaly type       can score well overall yet be blind to one category.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of true anomalies among the k highest-scored records."""
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(np.asarray(labels)[top_idx].mean())


def recall_at_k_per_type(
    labels: np.ndarray, types: list[str], scores: np.ndarray, k: int
) -> dict[str, float]:
    """Per anomaly type: fraction of its instances inside the top-k alerts."""
    top_idx = set(np.argsort(scores)[::-1][: min(k, len(scores))].tolist())
    result: dict[str, float] = {}
    for anomaly_type in sorted({t for t, y in zip(types, labels) if y and t}):
        member_idx = [i for i, (t, y) in enumerate(zip(types, labels)) if y and t == anomaly_type]
        found = sum(1 for i in member_idx if i in top_idx)
        result[anomaly_type] = found / len(member_idx)
    return result


def evaluate_detector(labels: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    """All headline metrics for one detector in one dict (one table row)."""
    labels = np.asarray(labels, dtype=int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        f"precision@{k}": precision_at_k(labels, scores, k),
    }
