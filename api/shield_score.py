"""POST /shield/score — fraud ML scoring for Finu Shield."""

import logging
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from shield.scoring import score_transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shield/score", tags=["shield"])


class ShieldScoreResponse(BaseModel):
    transaction_id: str
    model_available: bool
    model_source: str = Field(description="active | unavailable")
    model_name: str
    model_version: str
    isolation_score: float = 0.0
    behavioral_score: float = 0.0
    combined_score: float = 0.0
    feature_importances: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    total_latency_ms: float = 0.0


def _check_auth(x_internal_secret: Optional[str]) -> None:
    expected = os.environ.get("ML_INTERNAL_SECRET", "")
    if expected and x_internal_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("", response_model=ShieldScoreResponse)
async def shield_score(
    payload: dict[str, Any],
    x_internal_secret: Optional[str] = Header(default=None, alias="x-internal-secret"),
):
    _check_auth(x_internal_secret)
    t0 = time.time()
    try:
        result = score_transaction(payload)
        return ShieldScoreResponse(**result)
    except Exception as e:
        logger.exception("shield/score error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        logger.info("shield/score done in %.1fms", (time.time() - t0) * 1000)
