"""Anomaly detectors on embedding vectors.

All detectors are *unsupervised* — they never see the ground-truth labels.
Common interface: `fit_score(X)` returns one anomaly score per row, where
**higher = more anomalous**. Raw scores from different algorithms live on
different scales, so `normalize_scores` min-max-scales them into [0, 1]
for reporting; rank-based metrics (ROC-AUC, precision@k) are unaffected
by this monotone transformation.

Detectors:
  * IsolationForestDetector — isolates points by random axis-aligned
    splits; anomalies need fewer splits to isolate. Strong general-purpose
    baseline, near-linear runtime.
  * LOFDetector — Local Outlier Factor compares each point's local density
    with that of its neighbours; catches *local* anomalies that global
    methods miss.
  * (k-NN distance) — the third detector lives in VectorStore.
    knn_mean_distance, because it is computed by the vector database
    itself. The pipeline wraps it under the name "knn_qdrant".

This module deliberately imports only numpy + scikit-learn, so unit tests
run without torch or a running Qdrant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; constant input maps to all zeros."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - lo) / (hi - lo)


class Detector(ABC):
    """Unsupervised detector interface: higher score = more anomalous."""

    name: str = "base"

    @abstractmethod
    def fit_score(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return an anomaly score per row (raw scale)."""


class IsolationForestDetector(Detector):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        # Fixed random_state keeps benchmark runs reproducible.
        self._model = IsolationForest(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        self._model.fit(X)
        # score_samples: higher = more normal -> negate so higher = anomalous.
        return -self._model.score_samples(X)


class LOFDetector(Detector):
    name = "lof"

    def __init__(self, n_neighbors: int = 20):
        # novelty=False: score the training data itself (transductive),
        # which matches the batch-analysis setting of this project.
        self._model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, n_jobs=-1)

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        self._model.fit_predict(X)
        # negative_outlier_factor_: more negative = more anomalous -> negate.
        return -self._model.negative_outlier_factor_


DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    IsolationForestDetector.name: IsolationForestDetector,
    LOFDetector.name: LOFDetector,
}


# --------------------------------------------------------------------------- #
# Clustering for triage
# --------------------------------------------------------------------------- #

def cluster_embeddings(X: np.ndarray, eps: float = 0.35, min_samples: int = 5) -> np.ndarray:
    """DBSCAN over cosine distance; returns a cluster label per row (-1 = noise).

    Purpose: triage, not detection. Flagged anomalies that fall into the
    same cluster are very likely the *same kind* of incident (one OOM loop,
    one brute-force wave), so an on-call engineer reviews clusters instead
    of hundreds of individual lines. DBSCAN fits because the number of
    incident types is unknown up front and it has an explicit noise label.
    """
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X)


def summarize_clusters(labels: np.ndarray, X: np.ndarray, texts: list[str]) -> list[dict]:
    """Per-cluster summary with a representative sample (closest to the
    cluster centroid) — small, JSON-friendly dicts for the report."""
    summaries: list[dict] = []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        if label == -1:
            summaries.append({"cluster": -1, "size": int(idx.size), "representative": "<noise>"})
            continue
        centroid = X[idx].mean(axis=0)
        # Representative = member closest to the centroid.
        distances = np.linalg.norm(X[idx] - centroid, axis=1)
        rep_index = int(idx[int(np.argmin(distances))])
        summaries.append(
            {"cluster": int(label), "size": int(idx.size), "representative": texts[rep_index]}
        )
    return summaries
