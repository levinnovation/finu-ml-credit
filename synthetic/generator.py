"""Synthetic data generator for Credit Intelligence (Costa Rica).

Generates applicant-level rows matching the canonical `PERSONAL_CREDIT_V1`
feature schema (pipeline/schemas.py) -- the same schema used by
pipeline/features.py and the champion model registry (models/registry.py)
-- with two labels:
  - `elegible`: eligibility gate ("sujeto a credito")
  - `default_12m`: probability-of-default target

This is a Structural Causal Model (SCM)-style generator: each feature is
drawn from a distribution calibrated against PUBLIC AGGREGATE statistics
(INEC ENAHO 2025, BCCR/SUGEF 2025 delinquency indicators -- see
docs/data-schema-cr.md section 5), then `elegible` and `default_12m` are
derived through hand-specified causal rules + noise, NOT from real
individual bureau records (we have no access to those -- see
docs/data-schema-cr.md section 1).

Output is marked with `data_source = "synthetic_v1"` in every row.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.schemas import PERSONAL_CREDIT_V1

logger = logging.getLogger(__name__)

DATA_SOURCE_TAG = "synthetic_v1"

# ─── Public-data calibration anchors (see docs/data-schema-cr.md §5) ────────
INCOME_QUINTILE_MEANS_CRC = {
    1: 275_771,
    2: 560_500,
    3: 925_093,
    4: 1_391_137,
    5: 2_897_190,
}
QUINTILE_WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]
CONSUMER_CARD_DELINQUENCY_RATE = 0.031  # BCCR IAEF 2025, consumo/tarjetas

# Canonical feature order -- imported from the same schema the champion
# model registry serves with, so train/serve never drift.
CONTINUOUS_FEATURES = PERSONAL_CREDIT_V1.features
EMPLOYMENT_TYPES = ["asalariado", "independiente", "pensionado"]


@dataclass
class GeneratorConfig:
    n_rows: int = 20_000
    seed: int = 42
    default_base_rate: float = CONSUMER_CARD_DELINQUENCY_RATE


def _sample_income(rng: np.random.Generator, n: int) -> np.ndarray:
    quintiles = rng.choice([1, 2, 3, 4, 5], size=n, p=QUINTILE_WEIGHTS)
    means = np.array([INCOME_QUINTILE_MEANS_CRC[q] for q in quintiles])
    # log-normal noise around each quintile mean, household -> personal income approx 0.6x
    personal_income = means * 0.6 * rng.lognormal(mean=0.0, sigma=0.25, size=n)
    return np.clip(personal_income, 150_000, 20_000_000)


def generate(cfg: Optional[GeneratorConfig] = None) -> pd.DataFrame:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_rows

    income = _sample_income(rng, n)
    age = np.clip(rng.normal(38, 12, n), 18, 80).astype(int)
    employment_months = np.clip(rng.exponential(48, n), 0, 480)
    employment_type = rng.choice(EMPLOYMENT_TYPES, size=n, p=[0.65, 0.25, 0.10])
    employment_type_encoded = np.array([EMPLOYMENT_TYPES.index(t) for t in employment_type])

    active_debts = rng.poisson(1.8, n)

    # Bureau score proxy (300-850), correlated with income + history, plus noise
    equifax_score = (
        500
        + 60 * np.log1p(income / 500_000)
        + 0.4 * employment_months
        - 15 * active_debts
        + rng.normal(0, 60, n)
    )
    equifax_score = np.clip(equifax_score, 300, 850)

    delinquency_propensity = (850 - equifax_score) / 550 * 0.6 + np.clip(active_debts - 2, 0, None) * 0.05
    delinquency_propensity = np.clip(delinquency_propensity + rng.normal(0, 0.08, n), 0, 1)
    worst_delay_days = (rng.exponential(20, n) * delinquency_propensity * 3).round()

    credit_utilization_pct = np.clip(rng.beta(2, 3, n) * (1 + delinquency_propensity), 0, 1.5)
    avg_balance_3m = np.clip(income * rng.uniform(0.2, 1.5, n), 0, None)
    transaction_count_3m = rng.poisson(income / 50_000 + 10, n)

    loan_amount = income * rng.uniform(1.5, 12, n)
    term_months = rng.choice([12, 24, 36, 48, 60], size=n)
    monthly_payment = loan_amount / term_months
    dti_ratio = np.clip(monthly_payment / income, 0, 5)
    payment_to_income = np.clip((monthly_payment + active_debts * 50_000) / income, 0, 5)
    debt_service_coverage = np.clip(income / np.maximum(monthly_payment + active_debts * 50_000, 1), 0, 100)
    loan_to_income = np.clip(loan_amount / (income * 12), 0, 10)

    df = pd.DataFrame(
        {
            "age": age,
            "monthly_income": income,
            "employment_months": employment_months,
            "employment_type_encoded": employment_type_encoded,
            "equifax_score": equifax_score,
            "active_debts": active_debts,
            "worst_delay_days": worst_delay_days,
            "credit_utilization_pct": credit_utilization_pct,
            "avg_balance_3m": avg_balance_3m,
            "transaction_count_3m": transaction_count_3m,
            "dti_ratio": dti_ratio,
            "payment_to_income": payment_to_income,
            "debt_service_coverage": debt_service_coverage,
            "loan_to_income": loan_to_income,
            "employment_type": employment_type,
        }
    )
    assert list(df[CONTINUOUS_FEATURES].columns) == CONTINUOUS_FEATURES

    # ─── Eligibility gate (hard rules + soft noise) ─────────────────────────
    arrears_proxy = (worst_delay_days >= 30).astype(int) + (worst_delay_days >= 90).astype(int)
    hard_ineligible = (
        (df["age"] < 18)
        | (df["age"] > 75)
        | (df["employment_months"] < 1)
        | (arrears_proxy >= 2)
        | (worst_delay_days >= 90)
    )
    soft_eligibility_score = (
        0.5
        + 0.3 * (equifax_score - 575) / 275
        - 0.2 * delinquency_propensity
        + rng.normal(0, 0.1, n)
    )
    df["elegible"] = (~hard_ineligible) & (soft_eligibility_score > 0.35)

    # ─── Default label, base rate anchored to BCCR consumer/card delinquency ──
    logit = (
        np.log(cfg.default_base_rate / (1 - cfg.default_base_rate))
        + 1.8 * (dti_ratio - dti_ratio.mean())
        + 1.2 * (payment_to_income - payment_to_income.mean())
        - 0.015 * (equifax_score - equifax_score.mean())
        + 0.01 * worst_delay_days
        + rng.normal(0, 0.4, n)
    )
    proba_default = 1 / (1 + np.exp(-logit))
    df["default_12m"] = (rng.uniform(0, 1, n) < proba_default).astype(int)

    df["data_source"] = DATA_SOURCE_TAG
    df["dataset_version"] = "v1"
    df["feature_schema_version"] = PERSONAL_CREDIT_V1.version
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "v1"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = GeneratorConfig(n_rows=args.n_rows, seed=args.seed)
    df = generate(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "applicants.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    logger.info(f"elegible rate: {df['elegible'].mean():.3f}")
    logger.info(f"default_12m rate: {df['default_12m'].mean():.3f}")

    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    df.sample(n=min(200, len(df)), random_state=args.seed).to_csv(
        fixture_dir / "applicants_sample.csv", index=False
    )
    logger.info(f"Wrote fixture sample to {fixture_dir / 'applicants_sample.csv'}")


if __name__ == "__main__":
    main()
