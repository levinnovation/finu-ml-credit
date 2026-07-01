"""Regression tests for ml/explain.py.

Context: api/score.py used to report `feature_importances_` (identical for
every applicant) as "shap_value". That silently broke once the champion
became a `CalibratedClassifierCV` (used for probability calibration),
because calibrated wrappers don't expose `feature_importances_` at all --
`top_features` came back `[]` for every production request. See
ml/explain.py's module docstring.
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.explain import compute_shap_top_features

FEATURE_NAMES = ["age", "monthly_income", "dti_ratio", "active_debts"]


def _toy_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (X[:, 2] + X[:, 3] > 0).astype(int)  # depends on dti_ratio/active_debts
    return X, y


def test_calibrated_random_forest_returns_real_per_instance_shap():
    """This is the exact production shape: RandomForest wrapped in
    CalibratedClassifierCV. Before the fix, this silently returned []."""
    X, y = _toy_dataset()
    base = RandomForestClassifier(n_estimators=50, random_state=42, min_samples_leaf=5)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X, y)

    x0 = X[:1]
    feature_values = dict(zip(FEATURE_NAMES, x0[0].tolist()))
    top = compute_shap_top_features(calibrated, x0, FEATURE_NAMES, feature_values)

    assert top, "expected non-empty SHAP contributions for a calibrated tree model"
    assert len(top) <= 5
    for entry in top:
        assert entry["feature"] in FEATURE_NAMES
        assert isinstance(entry["shap_value"], float)
        assert entry["value"] == pytest.approx(feature_values[entry["feature"]])

    # SHAP should be able to distinguish two very different applicants.
    x1 = np.array([[0.0, 0.0, 5.0, 5.0]])  # extreme on the features that drive y
    top_extreme = compute_shap_top_features(
        calibrated, x1, FEATURE_NAMES, dict(zip(FEATURE_NAMES, x1[0].tolist()))
    )
    assert {e["feature"] for e in top_extreme} & {"dti_ratio", "active_debts"}


def test_plain_random_forest_returns_shap_without_calibration_wrapper():
    X, y = _toy_dataset()
    model = RandomForestClassifier(n_estimators=30, random_state=1)
    model.fit(X, y)

    x0 = X[:1]
    feature_values = dict(zip(FEATURE_NAMES, x0[0].tolist()))
    top = compute_shap_top_features(model, x0, FEATURE_NAMES, feature_values)
    assert top


def test_non_tree_estimator_degrades_to_empty_list_not_an_exception():
    """LogisticRegression isn't SHAP-tree-explainable here; must fail open
    (empty list) rather than crash the /score endpoint."""
    X, y = _toy_dataset()
    model = LogisticRegression().fit(X, y)
    top = compute_shap_top_features(model, X[:1], FEATURE_NAMES, {})
    assert top == []
