import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = FastAPI(title="FINU ML Credit", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "finu-ml-credit", "version": "0.1.0"}


@app.get("/models")
async def models():
    available = []
    try:
        import lightgbm
        available.append("lightgbm")
    except:
        pass
    try:
        import xgboost
        available.append("xgboost")
    except:
        pass
    return {"models": available, "status": "ok"}


@app.post("/score")
async def score(data: dict):
    return {"score": 0.5, "risk_band": "medium", "note": "placeholder"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
