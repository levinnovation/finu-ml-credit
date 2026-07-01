"""Synthetic data generator for Corporate Credit Intelligence (Costa Rica).

Generates company-level rows matching the canonical `CORPORATE_CREDIT_V1`
feature schema (pipeline/schemas.py), mirroring synthetic/generator.py's
approach for personal applicants: features are drawn from distributions
loosely calibrated to typical SME financial-statement ranges (no access to
real SUGEF/CIC corporate portfolios -- see docs/data-schema-cr.md), then
`default_12m` is derived through hand-specified causal rules + noise, NOT
from real corporate bureau records.

Output is marked with `data_source = "synthetic_corporate_v1"` in every row.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.schemas import CORPORATE_CREDIT_V1

logger = logging.getLogger(__name__)

DATA_SOURCE_TAG = "synthetic_corporate_v1"
CONTINUOUS_FEATURES = CORPORATE_CREDIT_V1.features

# Base annual default rate for CR commercial/SME lending, rough anchor
# (BCCR/SUGEF commercial portfolio delinquency indicators run lower than
# consumer -- see docs/data-schema-cr.md §5).
SME_DEFAULT_BASE_RATE = 0.045


@dataclass
class CorporateGeneratorConfig:
    n_rows: int = 10_000
    seed: int = 42
    default_base_rate: float = SME_DEFAULT_BASE_RATE


def generate(cfg: Optional[CorporateGeneratorConfig] = None) -> pd.DataFrame:
    cfg = cfg or CorporateGeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_rows

    # Monthly sales: log-normal, SME range roughly 2M-200M CRC/month
    monthly_sales = np.clip(rng.lognormal(mean=16.5, sigma=0.9, size=n), 1_000_000, 500_000_000)
    years_in_business = np.clip(rng.exponential(6, n), 0.1, 60)

    # EBITDA margin typically 5%-25% of annual sales, noisy
    ebitda_margin = np.clip(rng.normal(0.14, 0.08, n), -0.05, 0.4)
    ebitda_annual = monthly_sales * 12 * ebitda_margin

    current_ratio = np.clip(rng.lognormal(mean=0.15, sigma=0.5, size=n), 0.2, 8.0)
    debt_to_equity = np.clip(rng.lognormal(mean=0.1, sigma=0.7, size=n), 0.0, 15.0)

    active_debts = rng.poisson(1.2, n)
    delinquency_propensity = np.clip(
        0.5 * np.clip(debt_to_equity - 1.5, 0, None) / 5
        - 0.3 * np.clip(current_ratio - 1, 0, None)
        - 0.2 * ebitda_margin
        + rng.normal(0, 0.12, n),
        0,
        1,
    )
    worst_delay_days = (rng.exponential(15, n) * delinquency_propensity * 3).round()

    approx_annual_debt_service = np.maximum(monthly_sales * 12 * debt_to_equity * 0.05, 1.0)
    ebitda_coverage = np.clip(ebitda_annual / approx_annual_debt_service, -10, 50)

    loan_amount = monthly_sales * rng.uniform(1, 10, n)
    loan_to_sales = np.clip(loan_amount / np.maximum(monthly_sales * 12, 1), 0, 15)

    df = pd.DataFrame(
        {
            "monthly_sales": monthly_sales,
            "ebitda_annual": ebitda_annual,
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
            "ebitda_coverage": ebitda_coverage,
            "years_in_business": years_in_business,
            "loan_to_sales": loan_to_sales,
            "active_debts": active_debts,
            "worst_delay_days": worst_delay_days,
        }
    )
    assert list(df[CONTINUOUS_FEATURES].columns) == CONTINUOUS_FEATURES

    # ─── Default label, base rate anchored to SME commercial delinquency ────
    logit = (
        np.log(cfg.default_base_rate / (1 - cfg.default_base_rate))
        + 1.5 * (debt_to_equity - debt_to_equity.mean()) / max(debt_to_equity.std(), 1e-6)
        - 1.2 * (current_ratio - current_ratio.mean()) / max(current_ratio.std(), 1e-6)
        - 1.0 * (ebitda_coverage - ebitda_coverage.mean()) / max(ebitda_coverage.std(), 1e-6)
        - 0.4 * (years_in_business - years_in_business.mean()) / max(years_in_business.std(), 1e-6)
        + 0.01 * worst_delay_days
        + rng.normal(0, 0.4, n)
    )
    proba_default = 1 / (1 + np.exp(-logit))
    df["defaulted"] = (rng.uniform(0, 1, n) < proba_default).astype(int)

    df["data_source"] = DATA_SOURCE_TAG
    df["dataset_version"] = "v1"
    df["feature_schema_version"] = CORPORATE_CREDIT_V1.version
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic_corporate" / "v1"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = CorporateGeneratorConfig(n_rows=args.n_rows, seed=args.seed)
    df = generate(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "companies.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    logger.info(f"defaulted rate: {df['defaulted'].mean():.3f}")

    csv_path = out_dir / "companies.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote CSV to {csv_path}")

    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    df.sample(n=min(200, len(df)), random_state=args.seed).to_csv(
        fixture_dir / "companies_sample.csv", index=False
    )
    logger.info(f"Wrote fixture sample to {fixture_dir / 'companies_sample.csv'}")


if __name__ == "__main__":
    main()
