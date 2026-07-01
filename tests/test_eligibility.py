import numpy as np
from fastapi.testclient import TestClient

from main import app
from models.eligibility import EligibilityModel, hard_rule_check


def test_hard_rule_check_rejects_minors():
    passes, reasons = hard_rule_check({"age": 16, "employment_months": 12, "worst_delay_days": 0})
    assert not passes
    assert "age_below_minimum" in reasons


def test_hard_rule_check_passes_clean_applicant():
    passes, reasons = hard_rule_check({"age": 30, "employment_months": 24, "worst_delay_days": 0})
    assert passes
    assert reasons == []


def test_eligibility_model_without_classifier_reports_no_data_source():
    model = EligibilityModel()
    result = model.predict_one({"age": 30, "employment_months": 24, "worst_delay_days": 0})
    assert result["source"] == "hard_rule_fallback"
    assert result["data_source"] == "none"


def test_eligibility_model_fit_tags_data_source():
    rng = np.random.default_rng(42)
    X = rng.random((200, 3))
    y = (X[:, 0] > 0.5).astype(int)
    model = EligibilityModel().fit(X, y, ["a", "b", "c"], data_source="synthetic_v1")
    assert model.data_source == "synthetic_v1"

    result = model.predict_one({"age": 30, "employment_months": 24, "worst_delay_days": 0, "a": 0.9, "b": 0.1, "c": 0.1})
    assert result["source"] == "model"
    assert result["data_source"] == "synthetic_v1"


def test_eligibility_endpoint_fallback_reports_data_source_none(monkeypatch):
    import api.eligibility as eligibility_mod

    monkeypatch.setattr(eligibility_mod, "load_model", lambda name: None)
    monkeypatch.setattr(eligibility_mod, "_model", None)
    monkeypatch.setattr(eligibility_mod, "_model_available", False)

    client = TestClient(app)
    resp = client.post(
        "/eligibility",
        json={
            "tenant_id": "t-1",
            "cedula": "1-1111-1111",
            "application": {"age": 30, "employment_months": 24},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_available"] is False
    assert body["data_source"] == "none"
