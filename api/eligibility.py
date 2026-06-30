"""Eligibility ("sujeto a credito") gate endpoint. Runs before /score."""

import logging
import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from pipeline.features import compute_features
from models.storage import load_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eligibility", tags=["eligibility"])

_model = None
_model_available = False


def get_eligibility_model():
    global _model, _model_available
    if _model is not None:
        return _model
    from models.eligibility import EligibilityModel

    loaded = load_model("eligibility")
    if isinstance(loaded, EligibilityModel):
        _model = loaded
        _model_available = True
        logger.info("Eligibility model loaded")
    else:
        _model = EligibilityModel()  # hard-rules-only fallback
        _model_available = False
        logger.warning("No trained eligibility model found; serving hard-rules-only fallback")
    return _model


class EligibilityRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/workspace identifier")
    cedula: str = Field(..., description="Customer national ID")
    application: dict = Field(default_factory=dict)
    credit_data: Optional[dict] = Field(default=None)
    behavior_data: Optional[dict] = Field(default=None)


class EligibilityResponse(BaseModel):
    eligible: bool
    reasons: list[str]
    confidence: float
    source: str
    model_available: bool
    latency_ms: float


@router.post("", response_model=EligibilityResponse)
async def check_eligibility(request: EligibilityRequest):
    t0 = time.time()
    features = compute_features(
        request.application,
        credit_data=request.credit_data,
        behavior_data=request.behavior_data,
    )
    model = get_eligibility_model()
    result = model.predict_one(features)
    return EligibilityResponse(
        eligible=result["eligible"],
        reasons=result["reasons"],
        confidence=result["confidence"],
        source=result["source"],
        model_available=_model_available,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )
