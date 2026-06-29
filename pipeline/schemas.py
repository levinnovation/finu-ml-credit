"""Versioned feature schemas for train/serve parity."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    version: str
    features: List[str]
    monotone_constraints: Dict[str, int]


PERSONAL_CREDIT_V1 = FeatureSchema(
    name="credit_default_personal",
    version="personal_v1",
    features=[
        "age",
        "monthly_income",
        "employment_months",
        "employment_type_encoded",
        "equifax_score",
        "active_debts",
        "worst_delay_days",
        "credit_utilization_pct",
        "avg_balance_3m",
        "transaction_count_3m",
        "dti_ratio",
        "payment_to_income",
        "debt_service_coverage",
        "loan_to_income",
    ],
    monotone_constraints={
        "monthly_income": 1,
        "employment_months": 1,
        "equifax_score": 1,
        "active_debts": -1,
        "worst_delay_days": -1,
        "credit_utilization_pct": -1,
        "avg_balance_3m": 1,
        "transaction_count_3m": 1,
        "dti_ratio": -1,
        "payment_to_income": -1,
        "debt_service_coverage": 1,
        "loan_to_income": -1,
    },
)

CORPORATE_CREDIT_V1 = FeatureSchema(
    name="credit_default_corporate",
    version="corporate_v1",
    features=[
        "monthly_sales",
        "ebitda_annual",
        "current_ratio",
        "debt_to_equity",
        "ebitda_coverage",
        "years_in_business",
        "loan_to_sales",
        "active_debts",
        "worst_delay_days",
    ],
    monotone_constraints={
        "monthly_sales": 1,
        "ebitda_annual": 1,
        "current_ratio": 1,
        "debt_to_equity": -1,
        "ebitda_coverage": 1,
        "years_in_business": 1,
        "loan_to_sales": -1,
        "active_debts": -1,
        "worst_delay_days": -1,
    },
)

SCHEMAS = {
    PERSONAL_CREDIT_V1.version: PERSONAL_CREDIT_V1,
    CORPORATE_CREDIT_V1.version: CORPORATE_CREDIT_V1,
}


def get_schema(version: str = PERSONAL_CREDIT_V1.version) -> FeatureSchema:
    return SCHEMAS.get(version, PERSONAL_CREDIT_V1)
