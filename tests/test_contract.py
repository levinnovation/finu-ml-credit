import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def empty_registry_client(tmp_path, monkeypatch):
    reg = tmp_path / "model_registry.json"
    reg.write_text(json.dumps({"champion": None, "challenger": None, "models": []}))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(reg))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))
    # Re-import registry after env change
    import importlib
    import config
    import models.registry as registry_mod

    importlib.reload(config)
    importlib.reload(registry_mod)
    from main import app

    return TestClient(app)


def test_score_without_fitted_model_is_explicitly_unavailable(empty_registry_client):
    resp = empty_registry_client.post(
        "/score",
        json={
            "tenant_id": "tenant-1",
            "cedula": "1-0890-0456",
            "application": {
                "monthly_income": 900000,
                "amount": 3000000,
                "term_months": 36,
            },
            "credit_data": {
                "score": 680,
                "active_debts": 1,
                "worst_delay_days": 0,
            },
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model_available"] is False
    assert body["score"] is None
    assert body["probability_default"] is None
    assert body["risk_band"] == "unavailable"
    assert body["model_version"]
    assert body["score_0_100"] is None
    assert body["feature_schema_version"] == "personal_v1"
    assert "feature_values" in body


def test_health_reports_model_and_mlflow_metadata(empty_registry_client):
    resp = empty_registry_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "finu-ml-credit"
    assert "model_loaded" in body
    assert body["model_loaded"] is False
    assert "model_name" in body
    assert "model_version" in body
    assert "feature_schema_version" in body
    assert "mlflow_configured" in body


def test_models_reports_champion_contract(empty_registry_client):
    resp = empty_registry_client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_available" in body
    assert body["model_available"] is False
    assert "champion" in body
    assert "models" in body
    assert body["champion"]["loaded"] is False
    assert body["champion"]["feature_schema_version"] == "personal_v1"


def test_score_with_loaded_champion(tmp_path, monkeypatch):
    """Integration: champion registry produces model_available true and differentiated scores."""
    import importlib
    import pickle
    import shutil

    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    reg_dir = tmp_path / "models"
    reg_dir.mkdir()
    artifact = reg_dir / "champion.pkl"
    X = np.random.rand(200, 14)
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    with artifact.open("wb") as f:
        pickle.dump(model, f)

    manifest = {
        "champion": {
            "name": "credit_default_personal",
            "stage": "champion",
            "version": "test-v1",
            "model_type": "random_forest",
            "feature_schema_version": "personal_v1",
            "mlflow_run_id": None,
            "artifact_path": str(artifact),
            "metrics": {"roc_auc": 0.75},
            "thresholds": {"low_pd": 0.3, "medium_pd": 0.6},
        },
        "challenger": None,
        "models": [],
    }
    reg_path = reg_dir / "model_registry.json"
    reg_path.write_text(json.dumps(manifest))

    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(reg_path))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(reg_dir))

    import config
    import models.registry as registry_mod

    importlib.reload(config)
    importlib.reload(registry_mod)
    from main import app

    client = TestClient(app)
    resp = client.post(
        "/score",
        json={
            "tenant_id": "t-1",
            "cedula": "1-1111-1111",
            "application": {"monthly_income": 1_000_000, "amount": 2_000_000, "term_months": 36},
        },
    )
    body = resp.json()
    assert body["model_available"] is True
    assert body["score_0_100"] is not None
    assert body["probability_default"] is not None
