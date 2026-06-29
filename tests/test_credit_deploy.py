import io
import json
import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
import numpy as np


@pytest.fixture
def deploy_client(tmp_path, monkeypatch):
    reg_dir = tmp_path / "models"
    reg_dir.mkdir()
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(reg_dir / "model_registry.json"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(reg_dir))
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")

    import importlib
    import config
    import models.registry as registry_mod

    importlib.reload(config)
    importlib.reload(registry_mod)
    from main import app

    return TestClient(app), reg_dir


def test_deploy_registry_requires_auth(deploy_client):
    client, _ = deploy_client
    resp = client.post(
        "/credit/deploy-registry",
        headers={"x-cron-secret": "wrong-secret"},
        files={"registry": ("model_registry.json", b"{}", "application/json")},
    )
    assert resp.status_code == 401


def test_deploy_registry_writes_champion(deploy_client):
    client, reg_dir = deploy_client

    X = np.random.rand(50, 14)
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)
    artifact_name = "champion_deploy_test.pkl"
    artifact_path = reg_dir / artifact_name
    with artifact_path.open("wb") as f:
        pickle.dump(model, f)

    manifest = {
        "champion": {
            "name": "credit_default_personal",
            "stage": "champion",
            "version": "deploy-test-v1",
            "model_type": "random_forest",
            "feature_schema_version": "personal_v1",
            "artifact_path": artifact_name,
            "metrics": {"roc_auc": 0.75},
            "thresholds": {"low_pd": 0.3, "medium_pd": 0.6},
        },
        "challenger": None,
        "models": [],
    }

    reg_buf = io.BytesIO(json.dumps(manifest).encode())
    art_buf = io.BytesIO(artifact_path.read_bytes())

    resp = client.post(
        "/credit/deploy-registry",
        headers={"x-cron-secret": "test-cron-secret"},
        files={
            "registry": ("model_registry.json", reg_buf, "application/json"),
            "artifacts": (artifact_name, art_buf, "application/octet-stream"),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "deploy-test-v1"

    score = client.post(
        "/score",
        json={
            "tenant_id": "t",
            "cedula": "1-1111-1111",
            "application": {"monthly_income": 1_000_000, "amount": 2_000_000, "term_months": 36},
        },
    )
    assert score.json()["model_available"] is True
