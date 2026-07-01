"""Tests for the real-data eligibility retrain pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from credit import eligibility_retrain


@patch("credit.eligibility_retrain.count_labeled_decisions", return_value=50)
def test_eligibility_retrain_insufficient_labels(mock_count):
    result = eligibility_retrain.run_eligibility_retrain(min_labels=200, dry_run=True)
    assert result["skipped"] is True
    assert result["reason"] == "insufficient_labels"
    assert result["labels_available"] == 50
    mock_count.assert_called_once()


@patch("credit.eligibility_retrain.load_pg_eligibility_labels")
@patch("credit.eligibility_retrain.count_labeled_decisions", return_value=250)
def test_eligibility_retrain_dry_run(mock_count, mock_load):
    rng = np.random.default_rng(7)
    X = rng.random((250, 14))
    y = (rng.random(250) > 0.4).astype(int)
    mock_load.return_value = (X, y)

    result = eligibility_retrain.run_eligibility_retrain(min_labels=200, dry_run=True)
    assert result["skipped"] is False
    assert result["dry_run"] is True
    assert "metrics" in result
    assert "promotion" in result
    assert result["train_rows"] + result["test_rows"] == 250
