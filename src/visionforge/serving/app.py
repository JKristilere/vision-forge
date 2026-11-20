import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.visionforge.serving.predictor import InvalidImageError, Predictor
from src.visionforge.serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)
from src.visionforge.utils.logging import setup_logger

logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
ALLOWED_CONTENT_PREFIX = "image/"

# Environment variables that override configs/default.yaml "serving" settings (used by Docker).
ENV_OVERRIDES = {
    "VISIONFORGE_CHECKPOINT_PATH": ("checkpoint_path", str),
    "VISIONFORGE_DEVICE": ("device", str),
    "VISIONFORGE_TOP_K": ("top_k", int),
    "VISIONFORGE_MAX_UPLOAD_BYTES": ("max_upload_bytes", int),
}


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _apply_env_overrides(serving: dict) -> None:
    for env_name, (key, cast) in ENV_OVERRIDES.items():
        if env_name in os.environ:
            serving[key] = cast(os.environ[env_name])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The model is loaded once at startup and reused by every request.
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    _apply_env_overrides(config["serving"])
    app.state.max_upload_bytes = config["serving"]["max_upload_bytes"]
    app.state.predictor = None
    try:
        app.state.predictor = Predictor(config, _resolve_path(config["serving"]["checkpoint_path"]))
        logger.info("Model loaded on %s", app.state.predictor.device)
    except Exception:
        logger.exception("Failed to load model; /predict will return 503")
    yield


app = FastAPI(title="VisionForge Inference API", lifespan=lifespan)


def _get_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    if request.app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    return ModelInfoResponse(**_get_predictor(request).info())


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    predictor = _get_predictor(request)

    if not (file.content_type or "").startswith(ALLOWED_CONTENT_PREFIX):
        raise HTTPException(status_code=415, detail="Upload must be an image.")

    max_bytes = request.app.state.max_upload_bytes
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {max_bytes} bytes.")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Inference is CPU/GPU-bound, so run it off the event loop.
        result = await run_in_threadpool(predictor.predict_bytes, data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
