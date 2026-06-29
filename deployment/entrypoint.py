#!/usr/bin/env python3
"""Railway entrypoint for finu-ml-credit."""

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("finu-ml-credit.entrypoint")


def _try_startup_champion_sync() -> None:
    """If no champion on disk, attempt MLflow registry sync before serving traffic."""
    if os.environ.get("SKIP_STARTUP_MLFLOW_SYNC", "").lower() in ("1", "true", "yes"):
        return
    try:
        from config import settings
        from models.registry import get_champion

        if get_champion().loaded:
            logger.info("Champion already loaded — skip startup sync")
            return
        if not settings.mlflow_tracking_uri:
            logger.info("MLFLOW_TRACKING_URI unset — skip startup sync")
            return
        from mlflow_sync import download_champion_from_mlflow

        result = download_champion_from_mlflow()
        logger.info(
            "Startup MLflow sync OK: version=%s path=%s",
            result.get("model_version"),
            result.get("registry_path"),
        )
    except Exception as exc:
        logger.warning("Startup MLflow sync failed (cron/heal may recover): %s", exc)


PORT = os.environ.get("PORT", "8000")
_try_startup_champion_sync()
print(f"[finu-ml-credit] Starting FastAPI on :{PORT}", flush=True)
cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "main:app",
    "--host",
    "0.0.0.0",
    "--port",
    PORT,
    "--log-level",
    "info",
]
sys.exit(subprocess.call(cmd))
