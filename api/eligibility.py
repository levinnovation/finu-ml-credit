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


def get_eligibility_model():
    """Re-reads the cached pickle on every call (like models/registry.get_champion)
    rather than caching in a module global, so a retrain (training/train_eligibility_model.py
    or credit/eligibility_retrain.py) that overwrites the pickle takes effect on the next
    request without a process restart."""
    from models.eligibility import EligibilityModel

    loaded = load_model("eligibility")
    if isinstance(loaded, EligibilityModel):
        return loaded, True
    logger.warning("No trained eligibility model found; serving hard-rules-only fallback")
    return EligibilityModel(), False


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
    data_source: str = Field(
        default="none",
        description="Provenance of the eligibility model's training data: synthetic_v1 | production_decisions | none",
    )


@router.post("", response_model=EligibilityResponse)
async def check_eligibility(request: EligibilityRequest):
    t0 = time.time()
    features = compute_features(
        request.application,
        credit_data=request.credit_data,
        behavior_data=request.behavior_data,
    )
    model, model_available = get_eligibility_model()
    result = model.predict_one(features)
    return EligibilityResponse(
        eligible=result["eligible"],
        reasons=result["reasons"],
        confidence=result["confidence"],
        source=result["source"],
        model_available=model_available,
        latency_ms=round((time.time() - t0) * 1000, 1),
        data_source=result["data_source"],
    )
