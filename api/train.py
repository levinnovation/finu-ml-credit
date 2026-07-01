"""Ad hoc training endpoint — fine-tunes LightGBM / XGBoost with SUGEF 1-05
monotonicity constraints on a caller-supplied matrix.

IMPORTANT: this endpoint is NOT part of the production training pipeline.
The real champion model is produced by scripts/train_credit_default.py /
credit/retrain.py, which write to models/registry.py's manifest that
api/score.py actually reads. This endpoint fits a model and returns its
metrics/mlflow_run_id but never calls save_model/writes a registry entry --
it exists for one-off monotonicity experimentation (e.g. from a notebook or
this repo's own tests), not for promoting anything to serving. If you need
a caller-supplied-schema training path that actually reaches production,
extend credit/retrain.py instead of wiring this endpoint up.

Per the Credit Intelligence plan (Phase 2): models must enforce
monotone_constraints so a higher DTI (debt-to-income) or a worse delinquency
record can never *increase* the predicted credit score, even when noisy
data tempts the tree to fit a local non-monotonic pattern. This is
auditable by SUGEF reviewers.

Monotonicity signs are derived from pipeline/schemas.py's
FeatureSchema.monotone_constraints -- the SAME source of truth used by
ml/training_helpers.build_candidates() for the real training pipeline --
instead of a second, independently-maintained sign table. Any feature not
present in a known schema (e.g. a caller-supplied categorical) gets 0 (no
constraint).
"""

import logging
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import numpy as np

from pipeline.schemas import SCHEMAS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])


# ─── Monotonicity contract (SUGEF 1-05) ──────────────────────────────────────
#
# Sign convention: +1 = score should monotonically INCREASE as feature rises
#                  -1 = score should monotonically DECREASE as feature rises
#                   0 = no constraint (categorical, unused, or unknown feature)
#
# Merged from every known FeatureSchema (personal + corporate) so this
# endpoint stays in sync automatically if pipeline/schemas.py changes --
# no more hand-maintained duplicate table that can silently drift.
MONOTONE_FEATURE_SIGN: Dict[str, int] = {
    feature: sign
    for schema in SCHEMAS.values()
    for feature, sign in schema.monotone_constraints.items()
}

# Any feature that appears in a known schema's `features` list but NOT in its
# `monotone_constraints` dict is implicitly unconstrained (sign 0) -- no
# separate categorical table needed. Kept as an empty set for backwards
# compatibility with assert_no_categorical_leakage(); populate via
# `categorical_feature_names` in the request instead of a hardcoded list.
CATEGORICAL_FEATURES: set = set()


class TrainRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier")
    model_type: str = Field(default="lightgbm", description="lightgbm or xgboost")
    X: List = Field(..., description="Continuous feature matrix (categorical excluded)")
    y: List = Field(..., description="Labels (0=no default, 1=default)")
    feature_names: List[str] = Field(..., description="Column names matching X order")
    categorical_X: Optional[List] = Field(default=None, description="Categorical features (target-encoded or label-encoded)")
    categorical_feature_names: Optional[List[str]] = Field(default=None)
    experiment_name: str = Field(default="finu-credit-scoring")
    run_name: Optional[str] = None
    enforce_monotonicity: bool = Field(default=True, description="If False, skip the constraints tuple (debug only)")
    monotone_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="LightGBM monotone_constraints_method: 'advanced' with this strength")


class TrainResponse(BaseModel):
    status: str
    model_type: str
    samples: int
    features_continuous: int
    features_categorical: int
    default_rate: float
    training_time_ms: float
    monotone_applied: bool
    feature_signs: Dict[str, int]
    mlflow_run_id: Optional[str] = None


def build_monotone_constraints(
    feature_names: List[str],
    sign_table: Dict[str, int],
) -> List[int]:
    """Map the sign table onto the provided feature_names. Unmapped = 0 (no constraint)."""
    return [sign_table.get(name, 0) for name in feature_names]


@router.post("", response_model=TrainResponse)
async def train(request: TrainRequest):
    t0 = time.time()
    X = np.array(request.X, dtype=np.float64)
    y = np.array(request.y, dtype=np.int64)
    feature_names = list(request.feature_names)

    if len(X) < 10:
        raise HTTPException(400, "Need at least 10 samples to train")

    if not feature_names or len(feature_names) != X.shape[1]:
        raise HTTPException(400, "feature_names must match X column count")

    # Combine continuous + categorical into a single matrix for the trainer
    # (LightGBM can accept a single matrix with categorical_feature indices)
    if request.categorical_X is not None and request.categorical_feature_names:
        X_cat = np.array(request.categorical_X)
        cat_names = list(request.categorical_feature_names)
        if X_cat.shape[0] != X.shape[0]:
            raise HTTPException(400, "categorical_X row count must match X row count")
        X_full = np.hstack([X, X_cat])
        full_names = feature_names + cat_names
        cat_indices = list(range(len(feature_names), X_full.shape[1]))
    else:
        X_full = X
        full_names = feature_names
        cat_indices = []

    default_rate = float(y.mean())
    feature_signs = build_monotone_constraints(feature_names, MONOTONE_FEATURE_SIGN)
    monotone_applied = request.enforce_monotonicity and any(s != 0 for s in feature_signs)

    model = None
    if request.model_type == "lightgbm":
        import lightgbm as lgb
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "random_state": 42,
            "n_estimators": 100,
        }
        if monotone_applied:
            params["monotone_constraints"] = feature_signs
            # LightGBM ≥3.0 supports monotone_constraints_method for smoother
            # enforcement: 'advanced' allows the model to deviate locally
            # with a strength penalty (more auditable than 'basic').
            params["monotone_constraints_method"] = "advanced"
        model = lgb.LGBMClassifier(**params)
        if cat_indices:
            model.fit(X_full, y, categorical_feature=cat_indices)
        else:
            model.fit(X_full, y)
    elif request.model_type == "xgboost":
        import xgboost as xgb
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_estimators": 100,
        }
        if monotone_applied:
            # XGBoost expects monotone_constraints aligned to ALL columns.
            # Categorical features get 0 (no constraint) per the plan.
            full_signs = feature_signs + [0] * len(cat_indices)
            params["monotone_constraints"] = tuple(full_signs)
        if cat_indices:
            # XGBoost's sklearn API needs enable_categorical=True plus the
            # categorical columns as pandas `category` dtype -- it will NOT
            # infer them from plain float columns. Do this properly instead
            # of silently fitting as if they were continuous (that was the
            # previous behavior here: both branches called the exact same
            # `model.fit(X_full, y)`, so categorical columns were being fed
            # in as raw floats with no actual categorical handling).
            import pandas as pd

            params["enable_categorical"] = True
            model = xgb.XGBClassifier(**params)
            df_full = pd.DataFrame(X_full, columns=full_names)
            for cat_name in full_names[len(feature_names):]:
                # XGBoost's columnar categorical path requires int/string
                # category codes, not floats -- the request sends categorical
                # columns as JSON numbers, which numpy/pandas read as float64.
                df_full[cat_name] = df_full[cat_name].astype("int64").astype("category")
            model.fit(df_full, y)
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(X_full, y)
    else:
        raise HTTPException(400, f"Unknown model_type: {request.model_type}")

    run_id = None
    try:
        import mlflow
        mlflow.set_experiment(request.experiment_name)
        run_name = request.run_name or f"{request.model_type}-monotone-{int(time.time())}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", request.model_type)
            mlflow.log_param("samples", len(X))
            mlflow.log_param("features_continuous", X.shape[1])
            mlflow.log_param("features_categorical", len(cat_indices))
            mlflow.log_param("default_rate", default_rate)
            mlflow.log_param("monotone_applied", monotone_applied)
            if monotone_applied:
                # Log the sign per feature so SUGEF auditors can verify
                for fname, sign in zip(feature_names, feature_signs):
                    if sign != 0:
                        mlflow.set_tag(f"monotone_{fname}", sign)
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                pass
            run_id = mlflow.active_run().info.run_id
    except Exception as e:
        logger.info(f"MLflow logging skipped (not configured): {e}")

    training_time = (time.time() - t0) * 1000
    return TrainResponse(
        status="success",
        model_type=request.model_type,
        samples=len(X),
        features_continuous=X.shape[1],
        features_categorical=len(cat_indices),
        default_rate=round(default_rate, 4),
        training_time_ms=round(training_time, 1),
        monotone_applied=monotone_applied,
        feature_signs=dict(zip(feature_names, feature_signs)),
        mlflow_run_id=run_id,
    )
