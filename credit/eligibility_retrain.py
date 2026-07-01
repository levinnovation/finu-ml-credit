"""Eligibility retrain on real credit_decisions features — updates the
eligibility model cache the same way credit/retrain.py updates the default
model registry.

Unlike credit/retrain.py, this doesn't write to models/registry.py's
manifest (that's the default-risk champion). It saves via models/storage.py
under the "eligibility" name, same as training/train_eligibility_model.py's
synthetic path -- so api/eligibility.py's get_eligibility_model() picks it
up on next cold start with no other wiring changes needed.
"""

from __future__ import annotations

import os
import time
from typing import Any

from sklearn.model_selection import train_test_split

from config import settings
from credit.loader_pg import count_labeled_decisions, load_pg_eligibility_labels
from models.eligibility import EligibilityModel
from models.storage import save_model
from ml.metrics import compute_quality_metrics, passes_promotion_gate
from pipeline.schemas import PERSONAL_CREDIT_V1


def run_eligibility_retrain(
    min_labels: int | None = None,
    tenant_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    min_labels = min_labels or int(os.environ.get("CREDIT_RETRAIN_MIN_LABELS", "200"))
    new_count = count_labeled_decisions(tenant_id=tenant_id)

    if new_count < min_labels:
        return {
            "skipped": True,
            "reason": "insufficient_labels",
            "labels_available": new_count,
            "required": min_labels,
        }

    X, y = load_pg_eligibility_labels(min_rows=min_labels, tenant_id=tenant_id)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = EligibilityModel().fit(X_train, y_train, PERSONAL_CREDIT_V1.features, data_source="production_decisions")
    proba_test = model.classifier.predict_proba(X_test)[:, 1]
    metrics = compute_quality_metrics(y_test, proba_test).as_dict()
    ok, reason = passes_promotion_gate(metrics)

    if dry_run:
        return {
            "skipped": False,
            "dry_run": True,
            "metrics": metrics,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "promotion": {"accepted": ok, "reason": reason},
        }

    if not ok:
        return {"skipped": True, "reason": reason, "metrics": metrics}

    saved_path = save_model(model, "eligibility")

    run_id = None
    if settings.mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("finu-credit-eligibility")
            with mlflow.start_run(run_name=f"eligibility-production-{int(time.time())}") as run:
                mlflow.log_param("model_type", "lightgbm")
                mlflow.log_param("data_source", "production_decisions")
                mlflow.log_param("samples", int(X.shape[0]))
                for k, v in metrics.items():
                    if v is not None:
                        mlflow.log_metric(k, float(v))
                mlflow.sklearn.log_model(model.classifier, "model", serialization_format="pickle")
                run_id = run.info.run_id
        except Exception as exc:
            print(f"[mlflow] eligibility retrain logging skipped: {exc}")

    return {
        "skipped": False,
        "promoted": True,
        "metrics": metrics,
        "saved_path": saved_path,
        "train_rows": int(X_train.shape[0]),
        "mlflow_run_id": run_id,
        "data_source": "production_decisions",
    }
