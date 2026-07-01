"""POST /credit/eligibility-retrain — retrain the eligibility gate model from
real credit_decisions features (labeled via the deterministic hard-rule
check, see credit/eligibility_retrain.py). Mirrors api/credit_retrain.py's
auth pattern."""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from credit.eligibility_retrain import run_eligibility_retrain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/credit/eligibility-retrain", tags=["credit"])


class EligibilityRetrainResponse(BaseModel):
    skipped: bool
    reason: Optional[str] = None
    labels_available: Optional[int] = None
    required: Optional[int] = None
    promoted: Optional[bool] = None
    metrics: Optional[dict] = None
    saved_path: Optional[str] = None
    train_rows: Optional[int] = None
    test_rows: Optional[int] = None
    dry_run: Optional[bool] = None
    mlflow_run_id: Optional[str] = None
    data_source: Optional[str] = None


def _check_cron_auth(x_cron_secret: Optional[str]) -> None:
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("", response_model=EligibilityRetrainResponse)
async def eligibility_retrain(
    dry_run: bool = False,
    tenant_id: Optional[str] = Query(default=None),
    x_cron_secret: Optional[str] = Header(default=None, alias="x-cron-secret"),
):
    _check_cron_auth(x_cron_secret)
    try:
        result = run_eligibility_retrain(dry_run=dry_run, tenant_id=tenant_id)
        return EligibilityRetrainResponse(**result)
    except RuntimeError as e:
        logger.warning("credit/eligibility-retrain skipped: %s", e)
        return EligibilityRetrainResponse(skipped=True, reason=str(e))
    except Exception as e:
        logger.exception("credit/eligibility-retrain error")
        raise HTTPException(status_code=500, detail=str(e)) from e
