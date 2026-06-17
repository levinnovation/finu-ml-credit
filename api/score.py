"""Credit scoring API endpoint."""

import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pipeline.features import compute_features, to_array

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])

_ensemble = None


def get_ensemble():
    global _ensemble
    if _ensemble is not None:
        return _ensemble

    from models.ensemble import CreditEnsemble

    models = {}

    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        logger.info("LightGBM loaded")
    except Exception as e:
        logger.warning(f"LightGBM not available: {e}")

    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(n_estimators=100, random_state=42)
        logger.info("XGBoost loaded")
    except Exception as e:
        logger.warning(f"XGBoost not available: {e}")

    _ensemble = CreditEnsemble(models)
    return _ensemble


class ScoreRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/workspace identifier")
    cedula: str = Field(..., description="Customer national ID")
    application: dict = Field(default_factory=dict, description="Credit application fields")
    credit_data: Optional[dict] = Field(default=None, description="Credit bureau data (Equifax)")
    behavior_data: Optional[dict] = Field(default=None, description="Transaction behavior data")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="Blended default probability (0-1)")
    risk_band: str = Field(..., description="low | medium | high")
    model_count: int = Field(..., description="Number of models in ensemble")
    feature_count: int = Field(..., description="Number of features used")
    latency_ms: float = Field(..., description="Inference latency")


@router.post("", response_model=ScoreResponse)
async def score(request: ScoreRequest):
    t0 = time.time()

    features = compute_features(
        request.application,
        credit_data=request.credit_data,
        behavior_data=request.behavior_data,
    )
    X = to_array(features)

    ensemble = get_ensemble()
    try:
        proba = ensemble.predict_proba(X)[0]
    except Exception as e:
        logger.warning(f"Model not fitted, using fallback: {e}")
        risk_band = "medium"
        return ScoreResponse(
            score=0.5,
            risk_band=risk_band,
            model_count=len(ensemble.models),
            feature_count=len(features),
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    risk_band = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"

    latency_ms = (time.time() - t0) * 1000

    return ScoreResponse(
        score=round(float(proba), 4),
        risk_band=risk_band,
        model_count=len(ensemble.models),
        feature_count=len(features),
        latency_ms=round(latency_ms, 1),
    )
