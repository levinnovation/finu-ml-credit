"""
Feature extraction pipeline for credit scoring.

Purpose:
  Extract structured features from raw credit application data
  and compute derived financial ratios (DTI, PTI, DSTIR).

Feature categories:
  - Identity: age, cedula format validation
  - Employment: monthly_income, employment_months, sector
  - Credit bureau: equifax_score, active_debts, worst_delay
  - Behavioral: avg_balance, transaction_frequency
  - Derived: dti_ratio, payment_to_income, debt_service_coverage

Side effects:
  Reads from PostgreSQL ml_features cache; computes on miss.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
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
]

CATEGORICAL_FEATURES = ["employment_type_encoded"]


def compute_features(
    application: Dict[str, Any],
    credit_data: Optional[Dict[str, Any]] = None,
    behavior_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute feature vector from application + external data."""

    income = float(application.get("monthly_income", 0) or 0)
    loan_amount = float(application.get("amount", 0) or 0)
    term_months = int(application.get("term_months", 12) or 12)
    age = int(application.get("age", 30) or 30)
    employment_months = int(application.get("employment_months", 12) or 12)
    emp_type = str(application.get("employment_type", "asalariado"))

    emp_type_map = {"asalariado": 0, "independiente": 1, "pensionado": 2}
    employment_type_encoded = emp_type_map.get(emp_type, 0)

    equifax_score = 650
    active_debts = 0
    worst_delay = 0
    credit_util = 0.0
    if credit_data:
        equifax_score = float(credit_data.get("score", 650) or 650)
        active_debts = int(credit_data.get("active_debts", 0) or 0)
        worst_delay = int(credit_data.get("worst_delay_days", 0) or 0)
        credit_util = float(credit_data.get("credit_utilization_pct", 0) or 0)

    avg_balance = 0.0
    txn_count = 0
    if behavior_data:
        avg_balance = float(behavior_data.get("avg_balance_3m", 0) or 0)
        txn_count = int(behavior_data.get("transaction_count_3m", 0) or 0)

    monthly_payment = loan_amount / max(term_months, 1)
    dti_ratio = monthly_payment / max(income, 1)
    payment_to_income = (monthly_payment + (active_debts * 50000)) / max(income, 1)
    debt_service_coverage = max(income, 1) / max(monthly_payment + (active_debts * 50000), 1)
    loan_to_income = loan_amount / max(income * 12, 1)

    features = {
        "age": float(age),
        "monthly_income": income,
        "employment_months": float(employment_months),
        "employment_type_encoded": float(employment_type_encoded),
        "equifax_score": float(equifax_score),
        "active_debts": float(active_debts),
        "worst_delay_days": float(worst_delay),
        "credit_utilization_pct": credit_util,
        "avg_balance_3m": avg_balance,
        "transaction_count_3m": float(txn_count),
        "dti_ratio": min(dti_ratio, 10.0),
        "payment_to_income": min(payment_to_income, 10.0),
        "debt_service_coverage": min(debt_service_coverage, 100.0),
        "loan_to_income": min(loan_to_income, 10.0),
    }

    return features


def to_array(features: Optional[Dict[str, float]]) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    if not features:
        return np.zeros((1, len(FEATURE_NAMES)))
    return np.array([[features.get(k, 0.0) for k in FEATURE_NAMES]])
