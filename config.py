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

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
os.makedirs(settings.model_cache_dir, exist_ok=True)
