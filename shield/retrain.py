"""Shield retrain runner — trains LightGBM on pg labels and writes active_model.json."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from shield.loader_pg import count_labeled_feedback, load_pg_feedback
from shield.registry import reload_registry, shield_model_dir


def _last_retrain_path() -> Path:
    return shield_model_dir() / "last_retrain_at.txt"


def get_last_retrain_at() -> str | None:
    p = _last_retrain_path()
    if p.exists():
        return p.read_text().strip()
    return None


def run_retrain(
    min_feedback: int | None = None,
    tenant_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    min_feedback = min_feedback or int(os.environ.get("ML_RETRAIN_MIN_FEEDBACK", "50"))
    since = get_last_retrain_at()
    new_labels = count_labeled_feedback(since)

    if new_labels < min_feedback:
        return {
            "skipped": True,
            "reason": "insufficient_labels",
            "new_labels_since_last": new_labels,
            "required": min_feedback,
        }

    X, y = load_pg_feedback(min_rows=min_feedback, tenant_id=tenant_id or os.environ.get("DEFAULT_TENANT_ID"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError(f"lightgbm required for retrain: {e}") from e

    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
    }

    min_f1 = float(os.environ.get("SHIELD_PROMOTION_MIN_F1", "0.35"))
    if metrics["f1"] < min_f1:
        return {
            "skipped": True,
            "reason": "f1_below_minimum",
            "metrics": metrics,
            "required_f1": min_f1,
        }

    if dry_run:
        return {
            "skipped": False,
            "dry_run": True,
            "metrics": metrics,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        }

    out_dir = shield_model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "lightgbm"
    version = "v2"
    joblib_path = out_dir / f"{name}_{version}.joblib"
    meta_path = out_dir / f"{name}_{version}.metadata.json"
    joblib.dump(model, joblib_path)
    meta = {
        "name": name,
        "version": version,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_features": 14,
        "source": "pg_ml_feedback",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    active = {
        "name": name,
        "version": version,
        "joblib_path": joblib_path.name,
        "metrics": metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "active_model.json").write_text(json.dumps(active, indent=2))
    _last_retrain_path().write_text(datetime.now(timezone.utc).isoformat())
    reload_registry()

    return {
        "skipped": False,
        "promoted": name,
        "metrics": metrics,
        "model_dir": str(out_dir),
        "train_rows": int(X_train.shape[0]),
    }
