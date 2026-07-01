"""Corporate credit scoring: per-customer_type champion routing (Fase C).

Regression coverage for the bug this fixes: before this, /score always
resolved to the personal_v1 champion regardless of `customer_type`, and
corporate applications were silently scored with a zeroed-out personal
feature vector (see apps/fintech-saas's buildMlScorePayload, which never
populated corporate fields).
"""
import json
import pickle

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier


def _reload_registry(monkeypatch, reg_path, cache_dir):
    import importlib

    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(reg_path))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(cache_dir))
    import config
    import models.registry as registry_mod

    importlib.reload(config)
    importlib.reload(registry_mod)
    from main import app

    return TestClient(app)


@pytest.fixture
def dual_champion_client(tmp_path, monkeypatch):
    """A registry with BOTH a personal_v1 and a corporate_v1 champion,
    exercising the "champions" (keyed-by-model-name) manifest format."""
    from pipeline.schemas import CORPORATE_CREDIT_V1, PERSONAL_CREDIT_V1

    cache_dir = tmp_path / "models"
    cache_dir.mkdir()

    def _fit_and_dump(n_features: int, artifact_name: str):
        X = np.random.rand(200, n_features)
        y = (X[:, 0] > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        artifact = cache_dir / artifact_name
        with artifact.open("wb") as f:
            pickle.dump(model, f)
        return artifact

    personal_artifact = _fit_and_dump(len(PERSONAL_CREDIT_V1.features), "personal.pkl")
    corporate_artifact = _fit_and_dump(len(CORPORATE_CREDIT_V1.features), "corporate.pkl")

    manifest = {
        "champions": {
            PERSONAL_CREDIT_V1.name: {
                "name": PERSONAL_CREDIT_V1.name,
                "stage": "champion",
                "version": "personal-test-v1",
                "model_type": "random_forest",
                "feature_schema_version": PERSONAL_CREDIT_V1.version,
                "mlflow_run_id": None,
                "artifact_path": str(personal_artifact),
                "metrics": {"roc_auc": 0.7},
                "thresholds": {"low_pd": 0.3, "medium_pd": 0.6},
                "data_source": "synthetic_bootstrap",
            },
            CORPORATE_CREDIT_V1.name: {
                "name": CORPORATE_CREDIT_V1.name,
                "stage": "champion",
                "version": "corporate-test-v1",
                "model_type": "random_forest",
                "feature_schema_version": CORPORATE_CREDIT_V1.version,
                "mlflow_run_id": None,
                "artifact_path": str(corporate_artifact),
                "metrics": {"roc_auc": 0.68},
                "thresholds": {"low_pd": 0.3, "medium_pd": 0.6},
                "data_source": "synthetic_bootstrap",
            },
        },
        # Legacy singular keys kept in sync for personal only, mirroring
        # what scripts/train_credit_default.py writes.
        "champion": None,
        "challenger": None,
        "models": [],
    }
    reg_path = cache_dir / "model_registry.json"
    reg_path.write_text(json.dumps(manifest))

    return _reload_registry(monkeypatch, reg_path, cache_dir)


def test_score_defaults_to_personal_champion(dual_champion_client):
    resp = dual_champion_client.post(
        "/score",
        json={
            "tenant_id": "t-1",
            "cedula": "1-1111-1111",
            "application": {"monthly_income": 1_000_000, "amount": 2_000_000, "term_months": 36},
        },
    )
    body = resp.json()
    assert body["model_available"] is True
    assert body["model_name"] == "credit_default_personal"
    assert body["feature_schema_version"] == "personal_v1"


def test_score_with_customer_type_corporate_uses_corporate_champion(dual_champion_client):
    resp = dual_champion_client.post(
        "/score",
        json={
            "tenant_id": "t-1",
            "cedula": "3-101-123456",
            "customer_type": "corporate",
            "application": {
                "monthly_sales": 20_000_000,
                "ebitda_annual": 30_000_000,
                "current_assets": 15_000_000,
                "current_liabilities": 8_000_000,
                "total_assets": 50_000_000,
                "total_liabilities": 20_000_000,
                "years_in_business": 5,
                "amount": 10_000_000,
            },
        },
    )
    body = resp.json()
    assert body["model_available"] is True
    assert body["model_name"] == "credit_default_corporate"
    assert body["feature_schema_version"] == "corporate_v1"
    # Corporate feature vector, not the personal one -- this is the
    # regression case: a corporate application must never be scored with
    # zeroed personal features.
    assert set(body["feature_values"].keys()) == {
        "monthly_sales",
        "ebitda_annual",
        "current_ratio",
        "debt_to_equity",
        "ebitda_coverage",
        "years_in_business",
        "loan_to_sales",
        "active_debts",
        "worst_delay_days",
    }
    assert body["feature_values"]["monthly_sales"] == 20_000_000
    assert len(body["top_features"]) > 0
    for feat in body["top_features"]:
        assert feat["feature"] in body["feature_values"]


def test_score_corporate_without_champion_is_unavailable(tmp_path, monkeypatch):
    """When only a personal champion exists (pre-Fase-C state), corporate
    scoring must explicitly report unavailable -- not silently reuse the
    personal model."""
    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(json.dumps({"champion": None, "challenger": None, "models": []}))
    client = _reload_registry(monkeypatch, reg_path, tmp_path)

    resp = client.post(
        "/score",
        json={
            "tenant_id": "t-1",
            "cedula": "3-101-123456",
            "customer_type": "corporate",
            "application": {"monthly_sales": 10_000_000},
        },
    )
    body = resp.json()
    assert body["model_available"] is False
    assert body["model_name"] == "credit_default_corporate"
    assert body["feature_schema_version"] == "corporate_v1"
