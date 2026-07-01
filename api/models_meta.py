"""Model metadata endpoint."""

import logging
from fastapi import APIRouter
from config import settings
from models.registry import list_registry_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# Candidate model families actually built by ml/training_helpers.build_candidates()
# and considered for the champion by scripts/train_credit_default.py /
# credit/retrain.py. NOT an ensemble -- exactly ONE of these is selected as
# champion (see api/score.py: single `champion.estimator.predict_proba(X)`
# call, no blending). Kept here purely as informational metadata about what
# gets *trained*, distinct from `champion`/`challenger` below which report
# what's actually *serving*.
AVAILABLE_MODELS = [
    {"name": "Logistic Regression", "type": "linear", "framework": "sklearn", "zero_shot": False},
    {"name": "Random Forest", "type": "tree_ensemble", "framework": "sklearn", "zero_shot": False},
    {"name": "LightGBM", "type": "gradient_boosting", "framework": "lightgbm", "zero_shot": False},
    {"name": "XGBoost", "type": "gradient_boosting", "framework": "xgboost", "zero_shot": False},
]


@router.get("")
async def list_models():
    registry = list_registry_models()
    champion = registry["champion"]
    return {
        "active_version": champion["version"],
        "model_available": champion["loaded"],
        "champion": champion,
        "challenger": registry["challenger"],
        "mlflow_configured": bool(settings.mlflow_tracking_uri),
        "registry_path": registry["registry_path"],
        "models": AVAILABLE_MODELS,
        # Serving is single-model (the champion), not an ensemble -- this
        # used to be a hardcoded, disconnected {"lightgbm": 0.5, "xgboost": 0.5}
        # dict unrelated to what was actually selected. Report the real
        # champion's model_type with weight 1.0 instead of implying a blend
        # that doesn't exist in api/score.py.
        "ensemble_weights": {champion["model_type"]: 1.0} if champion["loaded"] else {},
    }
