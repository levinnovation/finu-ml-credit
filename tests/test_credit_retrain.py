"""Tests for credit retrain pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from credit import retrain as credit_retrain
from credit.loader_pg import LabelProvenance
from ml.metrics import passes_promotion_gate


def test_passes_promotion_gate_accepts_good_metrics():
    metrics = {"roc_auc": 0.72, "pr_auc": 0.35, "brier_score": 0.18}
    ok, reason = passes_promotion_gate(metrics, None)
    assert ok is True
    assert reason == "passed"


def test_passes_promotion_gate_rejects_low_auc():
    metrics = {"roc_auc": 0.55, "pr_auc": 0.35, "brier_score": 0.18}
    ok, reason = passes_promotion_gate(metrics, None)
    assert ok is False
    assert reason == "roc_auc_below_minimum"


@patch("credit.retrain.count_labeled_decisions", return_value=50)
def test_run_credit_retrain_insufficient_labels(mock_count):
    result = credit_retrain.run_credit_retrain(min_labels=200, dry_run=True)
    assert result["skipped"] is True
    assert result["reason"] == "insufficient_labels"
    mock_count.assert_called_once()


@patch("credit.retrain.get_champion")
@patch("credit.retrain.load_pg_labels")
@patch("credit.retrain.count_labeled_decisions", return_value=250)
def test_run_credit_retrain_dry_run(mock_count, mock_load, mock_champion):
    rng = np.random.default_rng(42)
    X = rng.random((250, 14))
    y = (rng.random(250) > 0.7).astype(int)
    provenance = LabelProvenance(verified_outcome=0, decision_proxy=250)
    mock_load.return_value = (X, y, provenance)
    champ = MagicMock()
    champ.loaded = False
    champ.metrics = None
    mock_champion.return_value = champ

    result = credit_retrain.run_credit_retrain(min_labels=200, dry_run=True)
    assert result["skipped"] is False
    assert result["dry_run"] is True
    assert "metrics" in result
    assert "promotion" in result
    assert result["label_provenance"] == {
        "label_verified_outcome_count": 0,
        "label_decision_proxy_count": 250,
        "label_verified_pct": 0.0,
    }


def test_label_provenance_graduates_when_fully_verified():
    all_verified = LabelProvenance(verified_outcome=250, decision_proxy=0)
    assert all_verified.verified_pct == 1.0
    mixed = LabelProvenance(verified_outcome=10, decision_proxy=240)
    assert mixed.verified_pct == 0.04
