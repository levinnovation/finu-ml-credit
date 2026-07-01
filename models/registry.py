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
from pipeline.schemas import CORPORATE_CREDIT_V1, PERSONAL_CREDIT_V1

# Default (empty-champion) feature schema version per model name, used when
# the manifest has no entry for that model at all.
_DEFAULT_SCHEMA_BY_MODEL = {
    PERSONAL_CREDIT_V1.name: PERSONAL_CREDIT_V1.version,
    CORPORATE_CREDIT_V1.name: CORPORATE_CREDIT_V1.version,
}


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
    # Provenance of the training data behind this model. Callers (api/score.py,
    # fintech-saas) use this to decide whether a decision should be labeled as
    # coming from a production-grade model or a synthetic placeholder.
    # Known values: "production_decisions" (real credit_decisions rows),
    # "synthetic_bootstrap" (scripts/bootstrap_champion.sh, gate was skipped
    # or passed on synthetic-only data), "unknown" (manifest predates this
    # field).
    data_source: str = "unknown"


def registry_path() -> Path:
    configured = getattr(settings, "model_registry_path", "")
    if configured:
        return Path(configured)
    return Path(settings.model_cache_dir) / "model_registry.json"


def empty_champion(model_name: str = PERSONAL_CREDIT_V1.name) -> RegistryModel:
    return RegistryModel(
        name=model_name,
        stage="champion",
        version=settings.model_version,
        model_type="unavailable",
        feature_schema_version=_DEFAULT_SCHEMA_BY_MODEL.get(model_name, PERSONAL_CREDIT_V1.version),
        mlflow_run_id=settings.mlflow_run_id or None,
        artifact_path=None,
        metrics={},
        thresholds={},
        loaded=False,
        estimator=None,
        data_source="none",
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
        data_source=raw.get("data_source", "unknown"),
    )


def get_champion(model_name: str = PERSONAL_CREDIT_V1.name) -> RegistryModel:
    """Look up the champion for `model_name` (e.g. "credit_default_personal"
    or "credit_default_corporate").

    Manifest formats supported (newest first):
      - {"champions": {"<model_name>": {...}, ...}, ...} -- per-type
        champions, written by scripts/train_credit_default.py --schema.
      - {"champion": {...}, ...} -- legacy single-champion format, predates
        corporate support. Treated as the personal_v1 champion only.
    """
    manifest = load_manifest()
    champions = manifest.get("champions")
    if isinstance(champions, dict) and model_name in champions:
        return _model_from_manifest(champions[model_name], "champion")
    if model_name == PERSONAL_CREDIT_V1.name:
        return _model_from_manifest(manifest.get("champion"), "champion")
    return empty_champion(model_name)


def get_challenger(model_name: str = PERSONAL_CREDIT_V1.name) -> Optional[RegistryModel]:
    manifest = load_manifest()
    challengers = manifest.get("challengers")
    if isinstance(challengers, dict):
        raw = challengers.get(model_name)
        return _model_from_manifest(raw, "challenger") if raw else None
    if model_name != PERSONAL_CREDIT_V1.name:
        return None
    raw = manifest.get("challenger")
    if not raw:
        return None
    return _model_from_manifest(raw, "challenger")


def _champion_summary(m: RegistryModel) -> Dict[str, Any]:
    return {
        "name": m.name,
        "stage": m.stage,
        "version": m.version,
        "model_type": m.model_type,
        "feature_schema_version": m.feature_schema_version,
        "mlflow_run_id": m.mlflow_run_id,
        "artifact_path": m.artifact_path,
        "metrics": m.metrics,
        "thresholds": m.thresholds,
        "loaded": m.loaded,
        "data_source": m.data_source,
    }


def list_registry_models() -> Dict[str, Any]:
    manifest = load_manifest()
    champion = get_champion()
    challenger = get_challenger()
    corporate_champion = get_champion(CORPORATE_CREDIT_V1.name)
    return {
        "registry_path": str(registry_path()),
        "champions_by_model": {
            PERSONAL_CREDIT_V1.name: _champion_summary(champion),
            CORPORATE_CREDIT_V1.name: _champion_summary(corporate_champion),
        },
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
            "data_source": champion.data_source,
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
            "data_source": challenger.data_source,
        },
        "models": manifest.get("models", []),
    }
