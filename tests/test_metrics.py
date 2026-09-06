"""Tests for the evaluation metrics."""

import numpy as np
import pytest

from loganomaly.evaluation.metrics import (
    evaluate_detector,
    precision_at_k,
    recall_at_k_per_type,
)


def test_precision_at_k_perfect_ranking():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])  # both anomalies on top
    assert precision_at_k(labels, scores, k=2) == 1.0
    assert precision_at_k(labels, scores, k=4) == 0.5


def test_precision_at_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        precision_at_k(np.array([1]), np.array([0.5]), k=0)


def test_recall_per_type_identifies_blind_spot():
    #             normal, oom,  oom,  sqli
    labels = np.array([0,    1,    1,    1])
    types = ["", "oom", "oom", "sqli_probe"]
    scores = np.array([0.0, 0.9, 0.8, 0.1])  # detector is blind to sqli
    recall = recall_at_k_per_type(labels, types, scores, k=2)
    assert recall["oom"] == 1.0
    assert recall["sqli_probe"] == 0.0


def test_evaluate_detector_returns_all_headline_metrics():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.3, 0.2, 0.99])
    result = evaluate_detector(labels, scores, k=1)
    assert result["roc_auc"] == 1.0       # perfect ranking
    assert result["pr_auc"] == 1.0
    assert result["precision@1"] == 1.0
