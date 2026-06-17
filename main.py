"""finu-ml-credit — FastAPI ML service for credit scoring."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.health import router as health_router
from api.score import router as score_router
from api.models_meta import router as models_router
from api.train import router as train_router

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} on port {settings.port}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="FINU ML Credit",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(score_router)
app.include_router(models_router)
app.include_router(train_router)
