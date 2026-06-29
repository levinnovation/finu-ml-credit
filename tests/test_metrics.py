"""Tests for promotion gate metrics."""

from ml.metrics import compute_quality_metrics, passes_promotion_gate
import numpy as np


def test_passes_promotion_gate_accepts_good_metrics():
    metrics = {
        "roc_auc": 0.75,
        "pr_auc": 0.35,
        "brier_score": 0.15,
    }
    ok, reason = passes_promotion_gate(metrics)
    assert ok is True
    assert reason == "passed"


def test_passes_promotion_gate_rejects_low_auc():
    metrics = {"roc_auc": 0.50, "pr_auc": 0.35, "brier_score": 0.15}
    ok, reason = passes_promotion_gate(metrics)
    assert ok is False
    assert reason == "roc_auc_below_minimum"


def test_compute_quality_metrics_shape():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_quality_metrics(y, p)
    d = m.as_dict()
    assert d["roc_auc"] is not None
    assert d["brier_score"] is not None
