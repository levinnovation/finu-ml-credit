# finu-ml-credit

ML scoring microservice for FINU credit intelligence platform.

Production scoring is registry-driven: `/score` only serves a loaded
champion model from `MODEL_REGISTRY_PATH` or
`/tmp/finu-models/model_registry.json`. If no champion artifact is
available, the service returns `model_available:false`; it does not
instantiate unfitted estimators or return neutral fallback scores.

## Models

- `credit_default_personal` champion: LightGBM monotonic calibrated recommended
- challengers: XGBoost, CatBoost, Logistic Regression
- `credit_default_corporate`, `income_consistency`, `document_risk` are planned model families

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models and ensemble weights |
| `/score` | POST | Score a credit application |
| `/train` | POST | Train a model (LightGBM or XGBoost) |

## Score Contract

`POST /score` returns:

- `model_available`
- `model_name`
- `model_version`
- `mlflow_run_id`
- `probability_default`
- `score_0_100`
- `risk_band`
- `feature_schema_version`
- `calibration_version`
- `decision_thresholds`
- `feature_values`
- `top_features`
- `latency_ms`

## Offline Training

```bash
python scripts/train_credit_default.py --dataset /path/to/personal_credit_dataset.csv
```

The dataset must include the `personal_v1` feature schema and a
`defaulted` label. The script trains candidates, calibrates probabilities,
computes ROC-AUC/PR-AUC/KS/Brier, applies a promotion gate, logs MLflow
metadata when configured, and writes the registry manifest.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | PostgreSQL connection (for feature store) |
| `MLFLOW_TRACKING_URI` | No | MLflow tracking server URL |
| `MLFLOW_S3_ENDPOINT_URL` | No | Minio/S3 endpoint for model artifacts |
| `AWS_ACCESS_KEY_ID` | No | Minio access key |
| `AWS_SECRET_ACCESS_KEY` | No | Minio secret key |
| `MODEL_STORAGE_BUCKET` | No | S3 bucket for model storage |
| `MODEL_REGISTRY_PATH` | No | Registry manifest path |
| `FEATURE_SCHEMA_VERSION` | No | Active feature schema version |
| `CALIBRATION_VERSION` | No | Calibration artifact/version |

## Deploy

Railway: Dockerfile + `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
