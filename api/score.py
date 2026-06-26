"""Credit scoring API endpoint."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from config import settings
from pipeline.features import compute_features, to_array
from pipeline.schemas import get_schema
from models.registry import get_champion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])

class ScoreRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/workspace identifier")
    cedula: str = Field(..., description="Customer national ID")
    application: dict = Field(default_factory=dict, description="Credit application fields")
    credit_data: Optional[dict] = Field(default=None, description="Credit bureau data (Equifax)")
    behavior_data: Optional[dict] = Field(default=None, description="Transaction behavior data")


class ScoreResponse(BaseModel):
    model_name: str = Field(..., description="Registered model name")
    score: Optional[float] = Field(default=None, description="Blended default probability (0-1)")
    probability_default: Optional[float] = Field(default=None, description="Blended default probability (0-1)")
    score_0_100: Optional[int] = Field(default=None, description="Credit score where higher is lower risk")
    risk_band: str = Field(..., description="low | medium | high")
    model_count: int = Field(..., description="Number of models in ensemble")
    feature_count: int = Field(..., description="Number of features used")
    latency_ms: float = Field(..., description="Inference latency")
    model_available: bool = Field(..., description="True only when an active fitted model produced the score")
    model_version: str = Field(..., description="Active model version")
    mlflow_run_id: Optional[str] = Field(default=None, description="MLflow run id for the active model")
    feature_schema_version: str = Field(..., description="Feature schema used by the active model")
    calibration_version: str = Field(..., description="Probability calibration version")
    decision_thresholds: dict = Field(default_factory=dict, description="Model/policy thresholds used for bands")
    feature_values: dict = Field(default_factory=dict, description="Exact feature vector used for serving")
    top_features: list = Field(default_factory=list, description="Top feature contributions when available")


@router.post("", response_model=ScoreResponse)
async def score(request: ScoreRequest):
    t0 = time.time()

    features = compute_features(
        request.application,
        credit_data=request.credit_data,
        behavior_data=request.behavior_data,
    )
    X = to_array(features)

    champion = get_champion()
    schema = get_schema(champion.feature_schema_version)
    if not champion.loaded:
        return ScoreResponse(
            model_name=champion.name,
            score=None,
            probability_default=None,
            score_0_100=None,
            risk_band="unavailable",
            model_count=0,
            feature_count=len(features),
            model_available=False,
            model_version=champion.version,
            mlflow_run_id=champion.mlflow_run_id,
            feature_schema_version=schema.version,
            calibration_version=settings.calibration_version,
            decision_thresholds=champion.thresholds,
            feature_values=features,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )
    try:
        raw = champion.estimator.predict_proba(X)
        proba = raw[:, 1][0] if getattr(raw, "ndim", 1) > 1 else raw[0]
    except Exception as e:
        logger.warning(f"Model not fitted or unavailable: {e}")
        return ScoreResponse(
            model_name=champion.name,
            score=None,
            probability_default=None,
            score_0_100=None,
            risk_band="unavailable",
            model_count=1,
            feature_count=len(features),
            model_available=False,
            model_version=champion.version,
            mlflow_run_id=champion.mlflow_run_id,
            feature_schema_version=schema.version,
            calibration_version=settings.calibration_version,
            decision_thresholds=champion.thresholds,
            feature_values=features,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    risk_band = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
    importances = {}
    if hasattr(champion.estimator, "feature_importances_"):
        importances = {
            schema.features[i] if i < len(schema.features) else str(i): float(v)
            for i, v in enumerate(champion.estimator.feature_importances_)
        }
    top_features = [
        {"feature": k, "value": features.get(k, None), "shap_value": v}
        for k, v in sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    ]

    latency_ms = (time.time() - t0) * 1000

    return ScoreResponse(
        model_name=champion.name,
        score=round(float(proba), 4),
        probability_default=round(float(proba), 4),
        score_0_100=round((1 - float(proba)) * 100),
        risk_band=risk_band,
        model_count=1,
        feature_count=len(features),
        model_available=True,
        model_version=champion.version,
        mlflow_run_id=champion.mlflow_run_id,
        feature_schema_version=schema.version,
        calibration_version=settings.calibration_version,
        decision_thresholds=champion.thresholds,
        feature_values=features,
        top_features=top_features,
        latency_ms=round(latency_ms, 1),
    )
