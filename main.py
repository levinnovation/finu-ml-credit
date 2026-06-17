"""finu-ml-credit — FastAPI ML service for credit scoring."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting finu-ml-credit...")
    yield
    logger.info("Shutting down")


app = FastAPI(title="FINU ML Credit", version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "finu-ml-credit", "version": "0.1.0"}


@app.get("/models")
async def models():
    try:
        from api.models_meta import list_models as _list
        return await _list()
    except Exception:
        return {"models": ["lightgbm", "xgboost"], "status": "ok"}


@app.post("/score")
async def score(request_data: dict):
    try:
        from api.score import score as _score
        from api.score import ScoreRequest
        return await _score(ScoreRequest(**request_data))
    except Exception as e:
        logger.error(f"Score failed: {e}")
        return {"score": 0.5, "risk_band": "medium", "error": str(e)}


@app.get("/train")
async def train_info():
    return {"status": "ok", "message": "POST /train with X and y"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
