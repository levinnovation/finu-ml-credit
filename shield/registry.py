"""Shield model registry — lazy joblib loader from volume."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib

from config import settings


@dataclass
class ModelEntry:
    name: str
    version: str
    model: Any
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""
    n_features: int = 14
    _joblib_path: Optional[Path] = None


_registry: dict[str, ModelEntry] = {}


def shield_model_dir() -> Path:
    configured = getattr(settings, "shield_model_dir", "") or ""
    if configured:
        return Path(configured)
    return Path(settings.model_cache_dir) / "shield" / "v2"


def _discover() -> None:
    global _registry
    _registry = {}
    v2_dir = shield_model_dir()
    if not v2_dir.exists():
        return
    for meta_path in v2_dir.glob("*_v2.metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"[shield/registry] failed to read {meta_path}: {e}")
            continue
        name = meta.get("name")
        version = meta.get("version", "v2")
        if not name:
            continue
        joblib_path = meta_path.with_name(f"{name}_{version}.joblib")
        if not joblib_path.exists():
            continue
        _registry[name] = ModelEntry(
            name=name,
            version=version,
            model=None,
            metrics=meta.get("metrics", {}),
            trained_at=meta.get("trained_at", ""),
            n_features=meta.get("n_features", 14),
            _joblib_path=joblib_path,
        )


def list_models() -> list[str]:
    if not _registry:
        _discover()
    return sorted(_registry.keys())


def get_model(name: str) -> Optional[ModelEntry]:
    if not _registry:
        _discover()
    entry = _registry.get(name)
    if entry is None:
        return None
    if entry.model is None and entry._joblib_path:
        try:
            entry.model = joblib.load(entry._joblib_path)
        except Exception as e:
            print(f"[shield/registry] load error {entry._joblib_path}: {e}")
            return None
    return entry


def get_active_model() -> Optional[ModelEntry]:
    if not _registry:
        _discover()
    active_path = shield_model_dir() / "active_model.json"
    if active_path.exists():
        try:
            active = json.loads(active_path.read_text())
            preferred = active.get("name")
            if preferred:
                m = get_model(preferred)
                if m is not None:
                    return m
        except Exception as e:
            print(f"[shield/registry] active_model.json error: {e}")
    for preferred in ("lightgbm", "xgboost", "logistic_regression", "iforest"):
        m = get_model(preferred)
        if m is not None:
            return m
    return None


def get_active_metadata() -> dict[str, Any]:
    active = get_active_model()
    active_path = shield_model_dir() / "active_model.json"
    manifest: dict[str, Any] = {}
    if active_path.exists():
        try:
            manifest = json.loads(active_path.read_text())
        except Exception:
            pass
    return {
        "models": [
            {
                "name": name,
                "version": e.version,
                "metrics": e.metrics,
                "trained_at": e.trained_at,
            }
            for name, e in _registry.items()
        ] if _registry else [],
        "active": {
            "name": active.name if active else None,
            "version": active.version if active else None,
            "metrics": active.metrics if active else {},
        },
        "manifest": manifest,
        "model_dir": str(shield_model_dir()),
    }


def reload_registry() -> None:
    global _registry
    _registry = {}
    _discover()
