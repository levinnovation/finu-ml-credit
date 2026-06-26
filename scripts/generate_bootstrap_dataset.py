"""Generate a synthetic personal_v1 CSV for bootstrap champion training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.schemas import PERSONAL_CREDIT_V1


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        income = float(rng.uniform(400_000, 3_000_000))
        amount = float(rng.uniform(500_000, 8_000_000))
        term = int(rng.integers(12, 84))
        monthly_payment = amount / max(term, 1)
        dti = monthly_payment / max(income, 1)
        equifax = float(rng.integers(450, 820))
        active_debts = int(rng.integers(0, 5))
        worst_delay = int(rng.integers(0, 12))
        emp_months = int(rng.integers(3, 120))
        age = int(rng.integers(22, 65))
        emp_type = int(rng.integers(0, 3))
        credit_util = float(rng.uniform(0, 0.95))
        avg_balance = float(rng.uniform(0, income * 2))
        txn_count = int(rng.integers(0, 80))
        pti = (monthly_payment + active_debts * 50_000) / max(income, 1)
        dsc = max(income, 1) / max(monthly_payment + active_debts * 50_000, 1)
        lti = amount / max(income * 12, 1)

        # Label driven by DTI + bureau score for bootstrap separability (E2E only)
        logit = (
            -3.2 * min(dti, 1.5)
            + 2.8 * (1 - equifax / 850)
            + 0.4 * min(worst_delay / 12, 1)
            + 0.3 * credit_util
            + rng.normal(0, 0.35)
        )
        prob_default = 1.0 / (1.0 + np.exp(-logit))
        defaulted = int(rng.random() < prob_default)

        rows.append({
            "age": age,
            "monthly_income": income,
            "employment_months": emp_months,
            "employment_type_encoded": emp_type,
            "equifax_score": equifax,
            "active_debts": active_debts,
            "worst_delay_days": worst_delay,
            "credit_utilization_pct": credit_util,
            "avg_balance_3m": avg_balance,
            "transaction_count_3m": txn_count,
            "dti_ratio": min(dti, 10),
            "payment_to_income": min(pti, 10),
            "debt_service_coverage": min(dsc, 100),
            "loan_to_income": min(lti, 10),
            "defaulted": defaulted,
        })

    df = pd.DataFrame(rows)
    missing = [c for c in PERSONAL_CREDIT_V1.features + ["defaulted"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/finu-models/bootstrap_personal_v1.csv")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out} (default rate {df['defaulted'].mean():.3f})")


if __name__ == "__main__":
    main()
