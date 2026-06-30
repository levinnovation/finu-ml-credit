"""Train the eligibility ("sujeto a credito") gate model on synthetic data
and optionally log it to MLflow.

Usage:
    python -m training.train_eligibility_model \
        --data data/synthetic/v1/applicants.parquet \
        --mlflow-uri http://localhost:5000

Loads the parquet produced by synthetic/generator.py, trains a LightGBM
classifier on the `elegible` label using the canonical PERSONAL_CREDIT_V1
feature schema (same one the champion default model uses, see
pipeline/schemas.py), and saves it via models/storage.py so api/eligibility.py
can load it with `load_model("eligibility")`.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from models.eligibility import EligibilityModel
from models.storage import save_model
from synthetic.generator import CONTINUOUS_FEATURES

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "finu-credit-eligibility"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/synthetic/v1/applicants.parquet")
    parser.add_argument("--mlflow-uri", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df = pd.read_parquet(args.data)
    feature_names = CONTINUOUS_FEATURES
    X = df[feature_names].values.astype(np.float64)
    y = df["elegible"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    t0 = time.time()
    model = EligibilityModel().fit(X_train, y_train, feature_names)
    training_time_ms = (time.time() - t0) * 1000

    proba_test = model.classifier.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, proba_test),
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
    }
    logger.info(f"Eligibility model metrics: {metrics}")

    saved_path = save_model(model, "eligibility")
    logger.info(f"Saved local model to {saved_path}")

    try:
        import mlflow

        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"eligibility-lightgbm-{int(time.time())}") as run:
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("samples", len(X))
            mlflow.log_param("data_source", "synthetic_v1")
            mlflow.log_param("feature_count", len(feature_names))
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_metric("training_time_ms", training_time_ms)
            mlflow.sklearn.log_model(model.classifier, "model", serialization_format="pickle")
            logger.info(f"Logged to MLflow run {run.info.run_id}")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
