"""Local/MLflow-backed model registry manifest.

The service never creates unfitted estimators for production scoring.
It loads a champion from a manifest. MLflow can write/sync the same
manifest after promotion; local cache keeps Railway startup simple.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from pipeline.schemas import PERSONAL_CREDIT_V1


@dataclass(frozen=True)
class RegistryModel:
    name: str
    stage: str
    version: str
    model_type: str
    feature_schema_version: str
    mlflow_run_id: Optional[str]
    artifact_path: Optional[str]
    metrics: Dict[str, Any]
    thresholds: Dict[str, Any]
    loaded: bool = False
    estimator: Any = None


def registry_path() -> Path:
    configured = getattr(settings, "model_registry_path", "")
    if configured:
        return Path(configured)
    return Path(settings.model_cache_dir) / "model_registry.json"


def empty_champion() -> RegistryModel:
    return RegistryModel(
        name="credit_default_personal",
        stage="champion",
        version=settings.model_version,
        model_type="unavailable",
        feature_schema_version=PERSONAL_CREDIT_V1.version,
        mlflow_run_id=settings.mlflow_run_id or None,
        artifact_path=None,
        metrics={},
        thresholds={},
        loaded=False,
        estimator=None,
    )


def load_manifest() -> Dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {"champion": None, "challenger": None, "models": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_artifact(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    artifact = Path(path)
    if artifact.is_absolute():
        return artifact
    return Path(settings.model_cache_dir) / artifact


def _load_estimator(path: Optional[str]) -> Any:
    artifact = _resolve_artifact(path)
    if not artifact or not artifact.exists():
        return None
    with artifact.open("rb") as f:
        return pickle.load(f)


def _model_from_manifest(raw: Optional[Dict[str, Any]], default_stage: str) -> RegistryModel:
    if not raw:
        return empty_champion() if default_stage == "champion" else RegistryModel(
            name="credit_default_personal",
            stage=default_stage,
            version="unavailable",
            model_type="unavailable",
            feature_schema_version=PERSONAL_CREDIT_V1.version,
            mlflow_run_id=None,
            artifact_path=None,
            metrics={},
            thresholds={},
        )
    estimator = _load_estimator(raw.get("artifact_path"))
    return RegistryModel(
        name=raw.get("name", "credit_default_personal"),
        stage=raw.get("stage", default_stage),
        version=raw.get("version", settings.model_version),
        model_type=raw.get("model_type", "unknown"),
        feature_schema_version=raw.get("feature_schema_version", PERSONAL_CREDIT_V1.version),
        mlflow_run_id=raw.get("mlflow_run_id") or settings.mlflow_run_id or None,
        artifact_path=raw.get("artifact_path"),
        metrics=raw.get("metrics", {}),
        thresholds=raw.get("thresholds", {}),
        loaded=estimator is not None,
        estimator=estimator,
    )


def get_champion() -> RegistryModel:
    return _model_from_manifest(load_manifest().get("champion"), "champion")


def get_challenger() -> Optional[RegistryModel]:
    raw = load_manifest().get("challenger")
    if not raw:
        return None
    return _model_from_manifest(raw, "challenger")


def list_registry_models() -> Dict[str, Any]:
    manifest = load_manifest()
    champion = get_champion()
    challenger = get_challenger()
    return {
        "registry_path": str(registry_path()),
        "champion": {
            "name": champion.name,
            "stage": champion.stage,
            "version": champion.version,
            "model_type": champion.model_type,
            "feature_schema_version": champion.feature_schema_version,
            "mlflow_run_id": champion.mlflow_run_id,
            "artifact_path": champion.artifact_path,
            "metrics": champion.metrics,
            "thresholds": champion.thresholds,
            "loaded": champion.loaded,
        },
        "challenger": None if challenger is None else {
            "name": challenger.name,
            "stage": challenger.stage,
            "version": challenger.version,
            "model_type": challenger.model_type,
            "feature_schema_version": challenger.feature_schema_version,
            "mlflow_run_id": challenger.mlflow_run_id,
            "artifact_path": challenger.artifact_path,
            "metrics": challenger.metrics,
            "thresholds": challenger.thresholds,
            "loaded": challenger.loaded,
        },
        "models": manifest.get("models", []),
    }
