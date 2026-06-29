from fastapi import APIRouter
from config import settings
from models.registry import get_champion
from shield.registry import get_active_model, shield_model_dir

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    champion = get_champion()
    shield_active = get_active_model()
    return {
        "status": "ok",
        "service": "finu-ml-credit",
        "version": "0.1.0",
        "model_loaded": champion.loaded,
        "model_name": champion.name,
        "model_version": champion.version,
        "feature_schema_version": champion.feature_schema_version,
        "calibration_version": settings.calibration_version,
        "mlflow_configured": bool(settings.mlflow_tracking_uri),
        "mlflow_run_id": champion.mlflow_run_id,
        "shield_model_loaded": shield_active is not None,
        "shield_model_name": shield_active.name if shield_active else None,
        "shield_model_dir": str(shield_model_dir()),
    }
