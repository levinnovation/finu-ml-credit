"""Tests for MLflow sync module (no live MLflow required)."""

from unittest.mock import MagicMock, patch

import pytest

from ml.metrics import passes_promotion_gate


def test_passes_promotion_gate_with_champion_margin():
    candidate = {"roc_auc": 0.70, "pr_auc": 0.25, "brier_score": 0.20}
    champion = {"roc_auc": 0.68}
    ok, reason = passes_promotion_gate(candidate, champion)
    assert ok is True


def test_download_champion_requires_mlflow_uri():
    from mlflow_sync import download_champion_from_mlflow

    with patch("mlflow_sync.settings") as mock_settings:
        mock_settings.mlflow_tracking_uri = ""
        with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
            download_champion_from_mlflow()


def test_download_champion_writes_manifest(tmp_path):
    from mlflow_sync import download_champion_from_mlflow

    fake_model = MagicMock()
    fake_run = MagicMock()
    fake_run.data.metrics = {"roc_auc": 0.72, "pr_auc": 0.30, "brier_score": 0.18}
    fake_run.data.params = {"model_type": "lightgbm", "feature_schema_version": "personal_v1"}

    fake_mv = MagicMock()
    fake_mv.run_id = "run-abc"
    fake_mv.version = "3"

    with patch("mlflow_sync.settings") as mock_settings:
        mock_settings.mlflow_tracking_uri = "http://mlflow.test"
        mock_settings.model_cache_dir = str(tmp_path)
        mock_settings.feature_schema_version = "personal_v1"
        mock_settings.mlflow_model_name = "credit_default_personal"
        mock_settings.mlflow_model_stage = "Production"

        with patch("mlflow_sync.registry_path", return_value=tmp_path / "model_registry.json"):
            with patch("mlflow.set_tracking_uri"), patch("mlflow.sklearn.load_model") as load_mock:
                import pickle
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression()
                model.fit([[1, 2], [3, 4]], [0, 1])

                def dump_model(*args, **kwargs):
                    p = tmp_path / "model.pkl"
                    with p.open("wb") as f:
                        pickle.dump(model, f)
                    return model

                load_mock.side_effect = dump_model

                with patch("mlflow.tracking.MlflowClient") as client_cls:
                    client = client_cls.return_value
                    client.get_latest_versions.return_value = [fake_mv]
                    client.get_run.return_value = fake_run
                    client.download_artifacts.side_effect = Exception("skip")

                    result = download_champion_from_mlflow()

    assert result["model_version"] == "3"
    assert (tmp_path / "model_registry.json").exists()
