"""Model metadata endpoint."""

import logging
from fastapi import APIRouter
from config import settings
from models.registry import list_registry_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

AVAILABLE_MODELS = [
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
        "ensemble_weights": {"lightgbm": 0.50, "xgboost": 0.50},
    }
