from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    app_name: str = "finu-ml-credit"
    debug: bool = False
    port: int = 8000

    database_url: str = Field(default="", alias="DATABASE_URL")
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    mlflow_s3_endpoint: str = Field(default="", alias="MLFLOW_S3_ENDPOINT_URL")
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")

    model_cache_dir: str = "/tmp/finu-models"
    model_bucket: str = Field(default="finu-models", alias="MODEL_STORAGE_BUCKET")
    model_version: str = Field(default="0.1.0", alias="MODEL_VERSION")
    model_registry_path: str = Field(default="", alias="MODEL_REGISTRY_PATH")
    feature_schema_version: str = Field(default="personal_v1", alias="FEATURE_SCHEMA_VERSION")
    calibration_version: str = Field(default="uncalibrated", alias="CALIBRATION_VERSION")
    mlflow_run_id: str = Field(default="", alias="MLFLOW_RUN_ID")

    shield_model_dir: str = Field(default="", alias="SHIELD_MODEL_DIR")
    ml_retrain_min_feedback: int = Field(default=50, alias="ML_RETRAIN_MIN_FEEDBACK")
    credit_retrain_min_labels: int = Field(default=200, alias="CREDIT_RETRAIN_MIN_LABELS")
    mlflow_model_name: str = Field(default="credit_default_personal", alias="MLFLOW_MODEL_NAME")
    mlflow_model_stage: str = Field(default="Production", alias="MLFLOW_MODEL_STAGE")
    cron_secret: str = Field(default="", alias="CRON_SECRET")
    ml_internal_secret: str = Field(default="", alias="ML_INTERNAL_SECRET")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
os.makedirs(settings.model_cache_dir, exist_ok=True)
