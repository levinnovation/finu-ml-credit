"""Credit retrain on credit_decisions labels — updates model_registry.json."""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from config import settings
from credit.loader_pg import count_labeled_decisions, load_pg_labels
from ml.metrics import compute_quality_metrics, passes_promotion_gate
from ml.training_helpers import build_candidates, maybe_log_mlflow
from models.registry import get_champion, registry_path
from pipeline.schemas import PERSONAL_CREDIT_V1


def _last_retrain_path() -> Path:
    return Path(settings.model_cache_dir) / "credit_last_retrain_at.txt"


def get_last_retrain_at() -> str | None:
    p = _last_retrain_path()
    return p.read_text().strip() if p.exists() else None


def run_credit_retrain(
    min_labels: int | None = None,
    tenant_id: str | None = None,
    dry_run: bool = False,
    source: str = "supabase",
) -> dict[str, Any]:
    min_labels = min_labels or int(os.environ.get("CREDIT_RETRAIN_MIN_LABELS", "200"))
    since = get_last_retrain_at()
    new_count = count_labeled_decisions(since=since, tenant_id=tenant_id)

    if source == "supabase" and new_count < min_labels:
        return {
            "skipped": True,
            "reason": "insufficient_labels",
            "new_labels_since_last": new_count,
            "required": min_labels,
        }

    X, y = load_pg_labels(min_rows=min_labels, tenant_id=tenant_id)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    champion = get_champion()
    champion_metrics = champion.metrics if champion.loaded else None

    best = None
    for name, estimator in build_candidates().items():
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_test)[:, 1]
        metrics = compute_quality_metrics(y_test, proba).as_dict()
        ok, reason = passes_promotion_gate(metrics, champion_metrics)
        candidate = {
            "name": PERSONAL_CREDIT_V1.name,
            "model_type": name,
            "metrics": metrics,
            "estimator": calibrated,
            "promotion": {"accepted": ok, "reason": reason},
        }
        if best is None or (metrics.get("roc_auc") or 0) > (best["metrics"].get("roc_auc") or 0):
            best = candidate

    if not best:
        return {"skipped": True, "reason": "no_candidates"}

    if dry_run:
        return {
            "skipped": False,
            "dry_run": True,
            "metrics": best["metrics"],
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "promotion": best["promotion"],
        }

    if not best["promotion"]["accepted"]:
        return {
            "skipped": True,
            "reason": best["promotion"]["reason"],
            "metrics": best["metrics"],
        }

    out_dir = Path(settings.model_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = f"{best['model_type']}-{int(time.time())}"
    artifact = out_dir / f"{PERSONAL_CREDIT_V1.name}_{version}.pkl"
    with artifact.open("wb") as f:
        pickle.dump(best["estimator"], f)

    run_id = maybe_log_mlflow(
        best["model_type"],
        best["estimator"],
        best["metrics"],
        f"credit_decisions:{X.shape[0]}",
        artifact,
    )

    entry = {
        "name": PERSONAL_CREDIT_V1.name,
        "stage": "champion",
        "version": version,
        "model_type": best["model_type"],
        "feature_schema_version": PERSONAL_CREDIT_V1.version,
        "mlflow_run_id": run_id,
        "artifact_path": artifact.name,
        "metrics": best["metrics"],
        "thresholds": {"low_pd": 0.30, "medium_pd": 0.60},
        "promotion": best["promotion"],
    }

    manifest = {"champion": entry, "challenger": None, "models": [entry]}
    reg = registry_path()
    tmp = reg.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(reg)
    _last_retrain_path().write_text(datetime.now(timezone.utc).isoformat())

    return {
        "skipped": False,
        "promoted": version,
        "metrics": best["metrics"],
        "registry_path": str(reg),
        "train_rows": int(X_train.shape[0]),
        "mlflow_run_id": run_id,
    }
