import numpy as np
from fastapi.testclient import TestClient

from main import app
from api.train import MONOTONE_FEATURE_SIGN
from pipeline.schemas import PERSONAL_CREDIT_V1

client = TestClient(app)


def test_monotone_signs_derived_from_schema_not_hardcoded():
    assert MONOTONE_FEATURE_SIGN["dti_ratio"] == PERSONAL_CREDIT_V1.monotone_constraints["dti_ratio"]
    assert MONOTONE_FEATURE_SIGN["monthly_income"] == PERSONAL_CREDIT_V1.monotone_constraints["monthly_income"]
    assert "age" not in MONOTONE_FEATURE_SIGN  # unconstrained -- absent, not 0-by-convention


def test_train_lightgbm_continuous_only():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2)).tolist()
    y = (rng.random(40) > 0.5).astype(int).tolist()
    resp = client.post(
        "/train",
        json={
            "tenant_id": "t-1",
            "model_type": "lightgbm",
            "X": X,
            "y": y,
            "feature_names": ["dti_ratio", "age"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["feature_signs"]["dti_ratio"] == -1


def test_train_xgboost_with_real_categorical_dtype_handling():
    rng = np.random.default_rng(1)
    X = rng.random((40, 1)).tolist()
    y = (rng.random(40) > 0.5).astype(int).tolist()
    cat_X = [[i % 3] for i in range(40)]
    resp = client.post(
        "/train",
        json={
            "tenant_id": "t-1",
            "model_type": "xgboost",
            "X": X,
            "y": y,
            "feature_names": ["dti_ratio"],
            "categorical_X": cat_X,
            "categorical_feature_names": ["provincia"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["features_categorical"] == 1
