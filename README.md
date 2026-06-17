# finu-ml-credit

ML scoring microservice for FINU credit intelligence platform.

## Models

- LightGBM (50% weight)
- XGBoost (50% weight)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models and ensemble weights |
| `/score` | POST | Score a credit application |
| `/train` | POST | Train a model (LightGBM or XGBoost) |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | PostgreSQL connection (for feature store) |
| `MLFLOW_TRACKING_URI` | No | MLflow tracking server URL |
| `MLFLOW_S3_ENDPOINT_URL` | No | Minio/S3 endpoint for model artifacts |
| `AWS_ACCESS_KEY_ID` | No | Minio access key |
| `AWS_SECRET_ACCESS_KEY` | No | Minio secret key |
| `MODEL_STORAGE_BUCKET` | No | S3 bucket for model storage |

## Deploy

Railway: Dockerfile + `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
