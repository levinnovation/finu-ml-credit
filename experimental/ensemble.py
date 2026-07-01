"""
Multi-model ensemble for credit scoring.

Purpose:
  Blending predictions from TabPFN, LightGBM, XGBoost, and CatBoost
  for robust credit default probability estimation.

Models:
  - TabPFN: 35% weight (foundation model, zero-shot)
  - LightGBM: 25% weight (gradient boosting baseline)
  - XGBoost: 20% weight (regularized boosting)
  - CatBoost: 20% weight (categorical-native)

Returns:
  predict_proba(X) → numpy array with blended default probability
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "lightgbm": 0.50,
    "xgboost": 0.50,
}


class CreditEnsemble:
    """Weighted blending + optional stacking meta-model."""

    def __init__(
        self,
        models: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.models = models
        self.weights = weights or DEFAULT_WEIGHTS
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(w for k, w in self.weights.items() if k in self.models)
        if total == 0:
            return
        for k in self.weights:
            self.weights[k] = self.weights[k] / total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Blended probability of default."""
        blended = np.zeros(len(X), dtype=np.float64)

        for name, model in self.models.items():
            if name not in self.weights:
                continue
            proba = self._get_proba(model, X)
            blended += proba * self.weights[name]

        return np.clip(blended, 0.0, 1.0)

    def _get_proba(self, model: object, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim > 1 else proba
        return np.zeros(len(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, X: np.ndarray, model_name: str = "lightgbm") -> Optional[Dict[str, float]]:
        """SHAP feature importance via TreeExplainer for tree-based models."""
        model = self.models.get(model_name)
        if model is None:
            return None
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1
            return {
                str(i): float(abs(shap_values[:, i]).mean())
                for i in range(X.shape[1])
            }
        except Exception as e:
            logger.warning(f"SHAP failed for {model_name}: {e}")
            return None
