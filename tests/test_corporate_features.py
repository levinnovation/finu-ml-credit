"""Corporate feature engineering (pipeline/features.py's compute_corporate_features)."""
from pipeline.features import compute_corporate_features
from pipeline.schemas import CORPORATE_CREDIT_V1


def test_compute_corporate_features_returns_exact_schema_keys():
    feat = compute_corporate_features(
        {
            "monthly_sales": 10_000_000,
            "ebitda_annual": 15_000_000,
            "current_assets": 8_000_000,
            "current_liabilities": 4_000_000,
            "total_assets": 30_000_000,
            "total_liabilities": 12_000_000,
            "years_in_business": 8,
            "amount": 5_000_000,
            "active_debts": 2,
            "worst_delay_days": 15,
        }
    )
    assert set(feat.keys()) == set(CORPORATE_CREDIT_V1.features)
    assert feat["current_ratio"] == 2.0
    assert feat["debt_to_equity"] == 12_000_000 / (30_000_000 - 12_000_000)
    assert feat["active_debts"] == 2.0
    assert feat["worst_delay_days"] == 15.0


def test_compute_corporate_features_handles_missing_fields_without_crashing():
    feat = compute_corporate_features({})
    assert set(feat.keys()) == set(CORPORATE_CREDIT_V1.features)
    for v in feat.values():
        assert v == 0.0
