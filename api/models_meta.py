"""Model metadata endpoint."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

AVAILABLE_MODELS = [
    {"name": "LightGBM", "type": "gradient_boosting", "framework": "lightgbm", "zero_shot": False},
    {"name": "XGBoost", "type": "gradient_boosting", "framework": "xgboost", "zero_shot": False},
]


@router.get("")
async def list_models():
    return {
        "active_version": "0.1.0",
        "models": AVAILABLE_MODELS,
        "ensemble_weights": {"lightgbm": 0.50, "xgboost": 0.50},
    }
