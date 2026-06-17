"""Model persistence — save/load to local cache directory.

For production, extend this to use Minio/S3 via MLflow artifact store.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


def save_model(model: object, name: str, version: str = "latest") -> str:
    """Save a fitted model to the local cache directory."""
    path = Path(settings.model_cache_dir) / f"{name}_{version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {path}")
    return str(path)


def load_model(name: str, version: str = "latest") -> Optional[object]:
    """Load a fitted model from the local cache directory."""
    path = Path(settings.model_cache_dir) / f"{name}_{version}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded: {path}")
    return model


def list_models() -> list[str]:
    """List all saved models."""
    cache = Path(settings.model_cache_dir)
    if not cache.exists():
        return []
    return [p.stem for p in cache.glob("*.pkl")]
