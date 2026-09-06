"""Tests for the synthetic log generator."""

import pytest

from loganomaly.models import load_jsonl, save_jsonl
from loganomaly.simulation.generator import ANOMALY_INJECTORS, generate_logs


def test_reproducible_with_seed():
    a = generate_logs(n=300, anomaly_rate=0.05, seed=7)
    b = generate_logs(n=300, anomaly_rate=0.05, seed=7)
    assert [r.message for r in a] == [r.message for r in b]


def test_different_seed_differs():
    a = generate_logs(n=300, seed=1)
    b = generate_logs(n=300, seed=2)
    assert [r.message for r in a] != [r.message for r in b]


def test_anomaly_rate_approximately_holds():
    records = generate_logs(n=5000, anomaly_rate=0.03, seed=42)
    rate = sum(r.is_anomaly for r in records) / len(records)
    assert 0.02 <= rate <= 0.04  # binomial wiggle room around 3%


def test_labels_are_consistent():
    records = generate_logs(n=1000, seed=3)
    for r in records:
        # anomaly_type is set exactly for anomalies, never for normals.
        assert bool(r.anomaly_type) == r.is_anomaly
        if r.is_anomaly:
            assert r.anomaly_type in ANOMALY_INJECTORS


def test_invalid_rate_rejected():
    with pytest.raises(ValueError):
        generate_logs(n=10, anomaly_rate=0.9)


def test_jsonl_roundtrip(tmp_path):
    records = generate_logs(n=50, seed=5)
    path = tmp_path / "logs.jsonl"
    save_jsonl(records, path)
    loaded = load_jsonl(path)
    assert loaded == records
