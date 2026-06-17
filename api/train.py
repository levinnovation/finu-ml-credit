"""Model training endpoint — fine-tunes models and logs to MLflow."""

import logging
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])


class TrainRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier")
    model_type: str = Field(default="lightgbm", description="Model to train: lightgbm, xgboost")
    X: list = Field(..., description="Feature matrix")
    y: list = Field(..., description="Labels (0=no default, 1=default)")
    feature_names: Optional[list] = Field(default=None)
    experiment_name: str = Field(default="finu-credit-scoring")
    run_name: Optional[str] = None


class TrainResponse(BaseModel):
    status: str
    model_type: str
    samples: int
    features: int
    default_rate: float
    training_time_ms: float
    mlflow_run_id: Optional[str] = None


@router.post("", response_model=TrainResponse)
async def train(request: TrainRequest):
    t0 = time.time()
    X = np.array(request.X)
    y = np.array(request.y)

    if len(X) < 10:
        raise HTTPException(400, "Need at least 10 samples to train")

    default_rate = float(y.mean())
    model = None

    if request.model_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    elif request.model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    else:
        raise HTTPException(400, f"Unknown model type: {request.model_type}")

    model.fit(X, y)

    try:
        import mlflow
        mlflow.set_experiment(request.experiment_name)
        run_name = request.run_name or f"{request.model_type}-{int(time.time())}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", request.model_type)
            mlflow.log_param("samples", len(X))
            mlflow.log_param("features", X.shape[1])
            mlflow.log_param("default_rate", default_rate)
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                pass
            run_id = mlflow.active_run().info.run_id
    except Exception as e:
        logger.info(f"MLflow logging skipped (not configured): {e}")
        run_id = None

    training_time = (time.time() - t0) * 1000
    return TrainResponse(
        status="success",
        model_type=request.model_type,
        samples=len(X),
        features=X.shape[1],
        default_rate=round(default_rate, 4),
        training_time_ms=round(training_time, 1),
        mlflow_run_id=run_id,
    )
