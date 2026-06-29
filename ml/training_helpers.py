"""Shared training helpers for credit default models."""

from __future__ import annotations

import time
from pathlib import Path

from config import settings
from pipeline.schemas import PERSONAL_CREDIT_V1


def build_candidates():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=42),
    }
    try:
        import lightgbm as lgb
        constraints = [PERSONAL_CREDIT_V1.monotone_constraints.get(f, 0) for f in PERSONAL_CREDIT_V1.features]
        candidates["lightgbm"] = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=300,
            random_state=42,
            verbose=-1,
            monotone_constraints=constraints,
        )
    except Exception:
        pass
    try:
        import xgboost as xgb
        constraints = tuple(PERSONAL_CREDIT_V1.monotone_constraints.get(f, 0) for f in PERSONAL_CREDIT_V1.features)
        candidates["xgboost"] = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=300,
            random_state=42,
            monotone_constraints=constraints,
        )
    except Exception:
        pass
    return candidates


def maybe_log_mlflow(
    name: str,
    model,
    metrics: dict,
    dataset_uri: str,
    artifact_path: Path,
    register: bool = True,
) -> str | None:
    if not settings.mlflow_tracking_uri:
        return None
    try:
        import mlflow
        import pandas as pd
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("finu-credit-default-personal")
        with mlflow.start_run(run_name=f"{name}-{int(time.time())}"):
            mlflow.log_params({
                "model_type": name,
                "feature_schema_version": PERSONAL_CREDIT_V1.version,
            })
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            mlflow.set_tag("dataset_uri", dataset_uri)
            mlflow.set_tag("model_name", PERSONAL_CREDIT_V1.name)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            signature = infer_signature(
                pd.DataFrame(columns=PERSONAL_CREDIT_V1.features),
                [0.1],
            )
            trusted_types = [
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration.CalibratedClassifierCV",
                "sklearn.linear_model._logistic.LogisticRegression",
                "sklearn.ensemble._forest.RandomForestClassifier",
            ]
            model_uri: str
            try:
                model_info = mlflow.sklearn.log_model(
                    model,
                    "sklearn_model",
                    signature=signature,
                    skops_trusted_types=trusted_types,
                )
                model_uri = model_info.model_uri
            except Exception:
                # Fallback: register from logged pickle artifact when sklearn flavor rejects types
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
            run_id = mlflow.active_run().info.run_id

            if register:
                model_name = getattr(settings, "mlflow_model_name", None) or PERSONAL_CREDIT_V1.name
                registered = mlflow.register_model(model_uri, model_name)
                stage = getattr(settings, "mlflow_model_stage", None) or "Production"
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered.version,
                    stage=stage,
                    archive_existing_versions=False,
                )
            return run_id
    except Exception as exc:
        print(f"[mlflow] skipped: {exc}")
        return None
