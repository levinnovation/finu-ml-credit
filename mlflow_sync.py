"""Sync champion model from MLflow Model Registry to local registry manifest."""

from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from ml.metrics import passes_promotion_gate
from models.registry import registry_path


def _model_name() -> str:
    return getattr(settings, "mlflow_model_name", None) or "credit_default_personal"


def _model_stage() -> str:
    return getattr(settings, "mlflow_model_stage", None) or "Production"


def download_champion_from_mlflow() -> Dict[str, Any]:
    if not settings.mlflow_tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not configured")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    name = _model_name()
    stage = _model_stage()
    versions = client.get_latest_versions(name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No MLflow model {name}@{stage}")

    mv = versions[0]
    run = client.get_run(mv.run_id)
    metrics = {k: float(v) for k, v in run.data.metrics.items() if v is not None}

    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{name}_{mv.version}_{int(time.time())}.pkl"
    local_path = cache_dir / artifact_name

    model_uri = f"models:/{name}/{stage}"
    try:
        loaded = mlflow.sklearn.load_model(model_uri)
        with local_path.open("wb") as f:
            pickle.dump(loaded, f)
    except Exception:
        # Fallback: download run artifacts
        artifact_path = client.download_artifacts(mv.run_id, "model", dst_path=str(cache_dir))
        pkl_candidates = list(Path(artifact_path).rglob("*.pkl"))
        if pkl_candidates:
            shutil.copyfile(pkl_candidates[0], local_path)
        else:
            raise RuntimeError(f"Could not download sklearn model from {model_uri}")

    champion_metrics = {
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "brier_score": metrics.get("brier_score"),
        "ks_statistic": metrics.get("ks_statistic"),
    }
    ok, reason = passes_promotion_gate(champion_metrics)
    if not ok:
        raise RuntimeError(f"MLflow champion failed promotion gate: {reason}")

    champion = {
        "name": name,
        "stage": "champion",
        "version": mv.version,
        "model_type": run.data.params.get("model_type", "mlflow_registry"),
        "feature_schema_version": run.data.params.get("feature_schema_version", settings.feature_schema_version),
        "mlflow_run_id": mv.run_id,
        "artifact_path": artifact_name,
        "metrics": champion_metrics,
        "thresholds": {"low_pd": 0.30, "medium_pd": 0.60},
        "promotion": {"accepted": True, "reason": "mlflow_registry_sync", "source": model_uri},
    }

    manifest = {"champion": champion, "challenger": None, "models": [champion]}
    reg_path = registry_path()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = reg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(reg_path)

    return {
        "registry_path": str(reg_path),
        "model_name": name,
        "model_version": mv.version,
        "mlflow_run_id": mv.run_id,
        "artifact_path": str(local_path),
        "metrics": champion_metrics,
    }
