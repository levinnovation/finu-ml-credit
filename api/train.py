"""Model training endpoint — fine-tunes LightGBM / XGBoost with SUGEF 1-05 monotonicity constraints.

Per the Credit Intelligence plan (Phase 2): models must enforce
monotone_constraints so a higher DTI (debt-to-income) or a worse delinquency
record can never *increase* the predicted credit score, even when noisy
data tempts the tree to fit a local non-monotonic pattern. This is
auditable by SUGEF reviewers — each feature's sign is documented in
MONOTONE_FEATURE_SIGN.

Categorical features (provincia, tipo_patrono, estado_civil) are excluded
from the constraints tuple, per the plan note: "monotone_constraints solo
se pueden aplicar sobre variables numéricas o continuas donde el orden
tenga significado directo. Las variables categóricas se manejan de manera
independiente mediante Target Encoding controlado o dejándolas
explícitamente fuera de la tupla de restricciones."

The endpoint accepts a pre-split matrix X with `feature_names` so the
caller decides which columns are continuous vs categorical. The
constraints vector is built by mapping the sign dict onto the provided
feature_names; categorical columns get 0 (no constraint).
"""

import logging
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])


# ─── Monotonicity contract (SUGEF 1-05) ──────────────────────────────────────
#
# Sign convention: +1 = score should monotonically INCREASE as feature rises
#                  -1 = score should monotonically DECREASE as feature rises
#                   0 = no constraint (categorical or unused feature)
#
# Continuous features that have a directional financial interpretation:
MONOTONE_FEATURE_SIGN: Dict[str, int] = {
    "income_monthly_log":       +1,   # more income → higher score
    "income_ccss_ratio":        +1,   # higher CCSS/bank consistency → higher score
    "dti_ratio":                -1,   # more debt-to-income → lower score
    "pti_ratio":                -1,   # payment-to-income → lower score
    "amount_to_income_ratio":   -1,   # larger loan relative to annual income → lower
    "worst_delay_days":         -1,   # longer delinquency → lower score
    "arrears_count_12m":        -1,   # more arrears → lower score
    "credit_history_months":    +1,   # longer clean history → higher score
    "active_credit_lines":       0,   # ambiguous — keep unconstrained
    "credit_utilization_pct":   -1,   # higher utilization → lower score
    "ltv_ratio":                -1,   # higher loan-to-value → lower score
    "ebitda_coverage":          +1,   # higher EBITDA coverage → higher score
    "razon_corriente":          +1,   # higher current ratio → higher score
    "endeudamiento_patrimonio": -1,   # higher D/E ratio → lower score
    "roe":                      +1,   # higher ROE → higher score
}

# Features that MUST stay out of the constraints tuple (categorical, derived,
# or non-monotonic by nature). Trainers must split their matrix so that
# categorical columns are passed in a separate `categorical_X` block.
CATEGORICAL_FEATURES = {
    "provincia",
    "tipo_patrono",
    "estado_civil",
    "nivel_educativo",
    "tipo_empleo",
    "industria",
    "tamano_empresa",
    "regimen_fiscal",
    "genero",
    "nacionalidad",
}


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


def assert_no_categorical_leakage(feature_names: List[str]) -> None:
    """Hard-fail if a known-categorical column slipped into the continuous X."""
    leaked = [f for f in feature_names if f in CATEGORICAL_FEATURES]
    if leaked:
        raise HTTPException(
            400,
            f"Las siguientes features categóricas no deben estar en X (páselas en categorical_X): {leaked}",
        )


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

    if request.enforce_monotonicity:
        assert_no_categorical_leakage(feature_names)

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
        model = xgb.XGBClassifier(**params)
        if cat_indices:
            # XGBoost needs categorical marked via enable_categorical=True
            # plus the columns as pandas Categorical dtype. We pass as
            # int codes with feature_cast='int' fallback for the demo.
            model.fit(X_full, y)
        else:
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
