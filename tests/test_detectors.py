"""Detector tests on synthetic numeric data (no embeddings needed).

Setup: a dense Gaussian blob of 'normal' points plus a handful of planted
far-away outliers. Any sane detector must rank the outliers on top.
"""

import numpy as np
import pytest

from loganomaly.detection.detectors import (
    DETECTOR_REGISTRY,
    cluster_embeddings,
    normalize_scores,
    summarize_clusters,
)


@pytest.fixture
def blob_with_outliers():
    """Mimics real (L2-normalised) embedding geometry: normal points share
    one direction with small perturbations, outliers point elsewhere.
    Vectors near the origin would make cosine distance meaningless."""
    rng = np.random.default_rng(0)
    dim = 16
    base = np.zeros(dim); base[0] = 1.0            # normal traffic direction
    away = np.zeros(dim); away[1] = 1.0            # outlier direction
    normal = base + rng.normal(scale=0.05, size=(300, dim))
    outliers = away + rng.normal(scale=0.05, size=(6, dim))
    X = np.vstack([normal, outliers]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # L2-normalise like the Embedder
    labels = np.array([0] * 300 + [1] * 6)
    return X, labels


@pytest.mark.parametrize("name", list(DETECTOR_REGISTRY))
def test_detectors_rank_planted_outliers_on_top(name, blob_with_outliers):
    X, labels = blob_with_outliers
    scores = DETECTOR_REGISTRY[name]().fit_score(X)
    top6 = np.argsort(scores)[::-1][:6]
    # All 6 planted outliers must occupy the top 6 ranks.
    assert set(top6.tolist()) == set(range(300, 306)), f"{name} missed outliers"


def test_normalize_scores_bounds():
    scores = normalize_scores(np.array([3.0, 5.0, 4.0]))
    assert scores.min() == 0.0 and scores.max() == 1.0


def test_normalize_constant_input_is_all_zero():
    assert (normalize_scores(np.ones(5)) == 0).all()


def test_dbscan_separates_blob_from_outliers(blob_with_outliers):
    X, _ = blob_with_outliers
    labels = cluster_embeddings(X, eps=0.35, min_samples=5)
    # The dense blob forms at least one real cluster ...
    assert (labels[:300] >= 0).mean() > 0.9
    # ... while the 6 far-away points are noise or a separate tiny cluster,
    # but never merged into a blob cluster.
    blob_clusters = set(labels[:300]) - {-1}
    assert not blob_clusters & set(labels[300:])


def test_cluster_summaries_have_representatives(blob_with_outliers):
    X, _ = blob_with_outliers
    texts = [f"line {i}" for i in range(len(X))]
    labels = cluster_embeddings(X)
    summaries = summarize_clusters(labels, X, texts)
    sizes = {s["cluster"]: s["size"] for s in summaries}
    assert sum(sizes.values()) == len(X)          # every point accounted for
    for s in summaries:
        if s["cluster"] != -1:
            assert s["representative"].startswith("line ")
