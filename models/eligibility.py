"""Eligibility gate ("sujeto a credito") — binary pre-screening model.

Runs BEFORE the default/risk scoring model (api/score.py / models/registry.py
champion). Combines hard deterministic rules (age, employment, severe
delinquency) with a learned LightGBM classifier for the soft eligibility
boundary, mirroring the rules+ML blend pattern already used app-side
(apps/fintech-saas/lib/credit/ml-client.ts: blendCreditScores).

Uses the same canonical feature schema as the champion default model
(pipeline/schemas.py: PERSONAL_CREDIT_V1) so the two models never drift.

Training data: services/finu-ml-credit/synthetic/generator.py
(see docs/data-schema-cr.md for calibration notes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Hard gates -- deterministic, never overridden by the model. Mirrors the
# `hard_ineligible` logic in synthetic/generator.py so train/serve stay
# consistent. Uses worst_delay_days (the only delinquency-history field in
# PERSONAL_CREDIT_V1) as a proxy for "excessive arrears" since this schema
# doesn't carry a separate 12-month arrears count.
MIN_AGE = 18
MAX_AGE = 75
MIN_EMPLOYMENT_MONTHS = 1
MAX_WORST_DELAY_DAYS = 90


def hard_rule_check(features: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Deterministic gates. Returns (passes, reason_codes_if_failed)."""
    reasons: List[str] = []
    age = features.get("age", 0)
    if age < MIN_AGE:
        reasons.append("age_below_minimum")
    if age > MAX_AGE:
        reasons.append("age_above_maximum")
    if features.get("employment_months", 0) < MIN_EMPLOYMENT_MONTHS:
        reasons.append("insufficient_employment_history")
    if features.get("worst_delay_days", 0) >= MAX_WORST_DELAY_DAYS:
        reasons.append("severe_delinquency")
    return (len(reasons) == 0, reasons)


class EligibilityModel:
    """Hard rules + LightGBM soft classifier for the eligibility gate."""

    def __init__(self, classifier: Optional[object] = None, threshold: float = 0.5):
        self.classifier = classifier
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        import lightgbm as lgb

        self.feature_names = feature_names
        self.classifier = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        self.classifier.fit(X, y)
        return self

    def predict_one(self, features: Dict[str, float]) -> Dict[str, Any]:
        passes_hard, hard_reasons = hard_rule_check(features)
        if not passes_hard:
            return {
                "eligible": False,
                "reasons": hard_reasons,
                "confidence": 1.0,
                "source": "hard_rule",
            }

        if self.classifier is None:
            # No trained model available -- conservative fallback: pass hard
            # rules only, flag as low-confidence so callers can decide.
            return {
                "eligible": True,
                "reasons": ["no_model_hard_rules_only"],
                "confidence": 0.5,
                "source": "hard_rule_fallback",
            }

        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        proba_eligible = float(self.classifier.predict_proba(X)[0][1])
        eligible = proba_eligible >= self.threshold
        reasons = [] if eligible else ["model_score_below_threshold"]
        return {
            "eligible": eligible,
            "reasons": reasons,
            "confidence": round(proba_eligible if eligible else 1 - proba_eligible, 4),
            "source": "model",
        }

    def save(self, path: str):
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "EligibilityModel":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
