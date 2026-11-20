# syntax=docker/dockerfile:1

# Stage 1: create /app/v_venv and install serving-only dependencies with pip
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
RUN python -m venv /app/v_venv
ENV PATH="/app/v_venv/bin:$PATH"

# CPU-only torch wheels from the PyTorch index avoid multi-GB CUDA packages.
COPY requirements-serve.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 --retries 10 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-serve.txt

# Stage 2: minimal runtime image with only the venv, config, and source code
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/v_venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VISIONFORGE_CHECKPOINT_PATH=artifacts/checkpoints/best.pth \
    VISIONFORGE_DEVICE=cpu \
    VISIONFORGE_TOP_K=3 \
    VISIONFORGE_MAX_UPLOAD_BYTES=5242880

RUN useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=builder /app/v_venv /app/v_venv
COPY configs ./configs
COPY src ./src

# Model weights are mounted at runtime (see docker-compose.yml), not baked into the image.
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "src.visionforge.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
