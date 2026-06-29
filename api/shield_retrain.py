"""POST /shield/retrain — nightly retrain on ml_feedback labels."""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from shield.retrain import run_retrain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shield/retrain", tags=["shield"])


class RetrainResponse(BaseModel):
    skipped: bool
    reason: Optional[str] = None
    new_labels_since_last: Optional[int] = None
    required: Optional[int] = None
    promoted: Optional[str] = None
    metrics: Optional[dict] = None
    model_dir: Optional[str] = None
    train_rows: Optional[int] = None
    dry_run: Optional[bool] = None


def _check_cron_auth(x_cron_secret: Optional[str]) -> None:
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("", response_model=RetrainResponse)
async def shield_retrain(
    dry_run: bool = False,
    x_cron_secret: Optional[str] = Header(default=None, alias="x-cron-secret"),
):
    _check_cron_auth(x_cron_secret)
    try:
        result = run_retrain(dry_run=dry_run)
        return RetrainResponse(**result)
    except RuntimeError as e:
        logger.warning("shield/retrain skipped: %s", e)
        return RetrainResponse(skipped=True, reason=str(e))
    except Exception as e:
        logger.exception("shield/retrain error")
        raise HTTPException(status_code=500, detail=str(e)) from e
