"""POST /credit/deploy-registry — upload champion registry + artifacts to the running service."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from config import settings
from models.registry import get_champion, registry_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/credit/deploy-registry", tags=["credit"])


def _check_cron_auth(x_cron_secret: Optional[str]) -> None:
    expected = os.environ.get("CRON_SECRET", "")
    if not expected or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("")
async def deploy_registry(
    registry: UploadFile = File(...),
    artifacts: List[UploadFile] = File(default=[]),
    x_cron_secret: Optional[str] = Header(default=None, alias="x-cron-secret"),
):
    """Write model_registry.json and .pkl artifacts into the service model cache."""
    _check_cron_auth(x_cron_secret)

    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    reg_bytes = await registry.read()
    try:
        manifest = json.loads(reg_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid registry JSON: {e}") from e

    for upload in artifacts:
        if not upload.filename or not upload.filename.endswith(".pkl"):
            raise HTTPException(status_code=400, detail=f"Invalid artifact: {upload.filename}")
        dest = cache_dir / Path(upload.filename).name
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        content = await upload.read()
        tmp.write_bytes(content)
        tmp.replace(dest)
        logger.info("deploy-registry wrote artifact %s (%d bytes)", dest.name, len(content))

    def _basename(entry: Optional[dict]) -> Optional[dict]:
        if not entry or not entry.get("artifact_path"):
            return entry
        fixed = dict(entry)
        fixed["artifact_path"] = Path(str(fixed["artifact_path"])).name
        return fixed

    manifest["champion"] = _basename(manifest.get("champion"))
    manifest["challenger"] = _basename(manifest.get("challenger"))
    manifest["models"] = [_basename(m) for m in manifest.get("models", [])]

    for key in ("champion", "challenger"):
        entry = manifest.get(key)
        if entry and entry.get("artifact_path"):
            p = cache_dir / entry["artifact_path"]
            if not p.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Registry references missing artifact: {entry['artifact_path']}",
                )

    reg_path = registry_path()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_reg = reg_path.with_suffix(".tmp")
    tmp_reg.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_reg.replace(reg_path)

    champion = get_champion()
    return {
        "status": "deployed",
        "registry_path": str(reg_path),
        "model_loaded": champion.loaded,
        "model_version": champion.version,
        "artifacts_written": len(artifacts),
    }
