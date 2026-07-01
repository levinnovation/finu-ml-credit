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


def to_array(features: Optional[Dict[str, float]], feature_names: Optional[list] = None) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order.

    `feature_names` defaults to the personal_v1 order for backward
    compatibility; pass CORPORATE_CREDIT_V1.features (or any schema's
    feature list) to serialize a different feature vector shape.
    """
    names = feature_names or FEATURE_NAMES
    if not features:
        return np.zeros((1, len(names)))
    return np.array([[features.get(k, 0.0) for k in names]])


def compute_corporate_features(application: Dict[str, Any]) -> Dict[str, float]:
    """Compute the corporate_v1 feature vector (see pipeline/schemas.py's
    CORPORATE_CREDIT_V1) from raw company financials.

    `application` fields (all optional, default to a conservative/neutral
    value so a partially-filled corporate application still scores rather
    than crashing):
      monthly_sales, ebitda_annual, current_assets, current_liabilities,
      total_assets, total_liabilities, years_in_business, amount,
      active_debts, worst_delay_days
    """
    monthly_sales = float(application.get("monthly_sales", 0) or 0)
    ebitda_annual = float(application.get("ebitda_annual", 0) or 0)
    current_assets = float(application.get("current_assets", 0) or 0)
    current_liabilities = float(application.get("current_liabilities", 0) or 0)
    total_assets = float(application.get("total_assets", 0) or 0)
    total_liabilities = float(application.get("total_liabilities", 0) or 0)
    years_in_business = float(application.get("years_in_business", 0) or 0)
    amount = float(application.get("amount", 0) or 0)
    active_debts = int(application.get("active_debts", 0) or 0)
    worst_delay_days = int(application.get("worst_delay_days", 0) or 0)

    equity = max(total_assets - total_liabilities, 1.0)
    current_ratio = current_assets / max(current_liabilities, 1.0)
    debt_to_equity = total_liabilities / equity
    # Approximate annual debt service as 20% of total liabilities (no
    # amortization schedule available at scoring time) -- a conservative
    # proxy so EBITDA coverage still moves sensibly with leverage.
    approx_annual_debt_service = max(total_liabilities * 0.2, 1.0)
    ebitda_coverage = ebitda_annual / approx_annual_debt_service
    loan_to_sales = amount / max(monthly_sales * 12, 1.0)

    return {
        "monthly_sales": monthly_sales,
        "ebitda_annual": ebitda_annual,
        "current_ratio": min(current_ratio, 50.0),
        "debt_to_equity": min(max(debt_to_equity, -50.0), 50.0),
        "ebitda_coverage": min(max(ebitda_coverage, -50.0), 50.0),
        "years_in_business": years_in_business,
        "loan_to_sales": min(loan_to_sales, 20.0),
        "active_debts": float(active_debts),
        "worst_delay_days": float(worst_delay_days),
    }
