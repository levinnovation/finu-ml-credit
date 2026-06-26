"""POST /credit/sync-mlflow — pull champion from MLflow Model Registry."""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from mlflow_sync import download_champion_from_mlflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/credit/sync-mlflow", tags=["credit"])


class SyncResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    registry_path: Optional[str] = None
    metrics: Optional[dict] = None


def _check_cron_auth(x_cron_secret: Optional[str]) -> None:
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("", response_model=SyncResponse)
async def credit_sync_mlflow(
    x_cron_secret: Optional[str] = Header(default=None, alias="x-cron-secret"),
):
    _check_cron_auth(x_cron_secret)
    try:
        result = download_champion_from_mlflow()
        return SyncResponse(status="synced", **result)
    except RuntimeError as e:
        logger.warning("credit/sync-mlflow skipped: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("credit/sync-mlflow error")
        raise HTTPException(status_code=500, detail=str(e)) from e
