"""POST /credit/retrain — retrain credit champion from credit_decisions labels."""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from credit.retrain import run_credit_retrain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/credit/retrain", tags=["credit"])


class CreditRetrainResponse(BaseModel):
    skipped: bool
    reason: Optional[str] = None
    new_labels_since_last: Optional[int] = None
    required: Optional[int] = None
    promoted: Optional[str] = None
    metrics: Optional[dict] = None
    registry_path: Optional[str] = None
    train_rows: Optional[int] = None
    dry_run: Optional[bool] = None
    mlflow_run_id: Optional[str] = None
    data_source: Optional[str] = None
    label_provenance: Optional[dict] = None


def _check_cron_auth(x_cron_secret: Optional[str]) -> None:
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("", response_model=CreditRetrainResponse)
async def credit_retrain(
    dry_run: bool = False,
    source: str = Query(default="supabase", description="supabase labels from credit_decisions"),
    customer_type: str = Query(default="personal", description="personal | corporate"),
    x_cron_secret: Optional[str] = Header(default=None, alias="x-cron-secret"),
):
    _check_cron_auth(x_cron_secret)
    try:
        result = run_credit_retrain(dry_run=dry_run, source=source, customer_type=customer_type)
        return CreditRetrainResponse(**result)
    except RuntimeError as e:
        logger.warning("credit/retrain skipped: %s", e)
        return CreditRetrainResponse(skipped=True, reason=str(e))
    except Exception as e:
        logger.exception("credit/retrain error")
        raise HTTPException(status_code=500, detail=str(e)) from e
