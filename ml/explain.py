"""Real per-instance SHAP explainability for served predictions.

api/score.py used to report `feature_importances_` (a GLOBAL, model-level
statistic identical for every applicant) mislabeled as "shap_value". That
silently broke down entirely once the champion became a
`CalibratedClassifierCV` (used for probability calibration): calibrated
wrappers don't expose `feature_importances_` at all, so `top_features` came
back empty for every single request in production.

This module computes genuine per-instance SHAP values -- the actual
contribution of *this applicant's* feature values to *this* prediction --
using shap.TreeExplainer against the underlying tree estimator(s), unwrapping
sklearn's CalibratedClassifierCV when present (averaging SHAP across its CV
folds, since each fold has its own fitted base estimator).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Estimators shap.TreeExplainer supports directly.
_TREE_MODEL_MODULES = (
    "sklearn.ensemble",
    "sklearn.tree",
    "lightgbm",
    "xgboost",
)


def _is_tree_model(estimator: Any) -> bool:
    module = type(estimator).__module__
    return any(module.startswith(prefix) for prefix in _TREE_MODEL_MODULES)


def _unwrap_base_estimators(estimator: Any) -> list[Any]:
    """Return the list of fitted tree estimators backing `estimator`.

    Handles sklearn's CalibratedClassifierCV (one base estimator per CV
    fold) transparently; falls back to `[estimator]` for a plain tree model.
    """
    calibrated_classifiers = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        bases = []
        for cc in calibrated_classifiers:
            base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if base is not None:
                bases.append(base)
        if bases:
            return bases
    return [estimator]


def compute_shap_top_features(
    estimator: Any,
    X: np.ndarray,
    feature_names: Sequence[str],
    feature_values: dict,
    top_k: int = 5,
) -> list[dict]:
    """Compute real per-instance SHAP contributions for one applicant.

    Returns [] (never raises) if the estimator isn't SHAP-explainable so
    callers can degrade gracefully -- an empty explanation is honest,
    unlike silently returning stale global importances.
    """
    try:
        import shap  # local import: heavy, training-only dep on some deployments
    except ImportError:
        logger.warning("shap not installed; top_features will be empty")
        return []

    bases = _unwrap_base_estimators(estimator)
    if not all(_is_tree_model(b) for b in bases):
        logger.info(
            "Champion estimator %s is not SHAP-tree-explainable; skipping top_features",
            type(estimator).__name__,
        )
        return []

    try:
        per_fold_shap = []
        for base in bases:
            explainer = shap.TreeExplainer(base)
            raw = explainer.shap_values(X)
            # Binary classifiers: shap_values is either a list [class0, class1]
            # or a single (n_samples, n_features) array for the positive class
            # depending on shap version/model type -- normalize to "class 1".
            if isinstance(raw, list):
                values = raw[1] if len(raw) > 1 else raw[0]
            else:
                values = raw
            values = np.asarray(values)
            if values.ndim == 3:
                # (n_samples, n_features, n_classes) in newer shap releases
                values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
            per_fold_shap.append(values[0])

        avg_shap = np.mean(per_fold_shap, axis=0)
    except Exception as e:
        logger.warning(f"SHAP computation failed, degrading to empty top_features: {e}")
        return []

    contributions = sorted(
        zip(feature_names, avg_shap),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:top_k]

    return [
        {
            "feature": name,
            "value": feature_values.get(name),
            "shap_value": round(float(value), 6),
        }
        for name, value in contributions
    ]
