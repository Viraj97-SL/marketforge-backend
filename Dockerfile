# MarketForge AI Backend — Railway Production Dockerfile
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure local src/ takes precedence over pip-installed marketforge-ai package
ENV PYTHONPATH=/app/src

# Copy deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time (avoids cold-start delays)
RUN python -m spacy download en_core_web_sm || echo "[WARN] spaCy unavailable"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "[WARN] SBERT unavailable"

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Bootstrap DB schema (idempotent)
RUN python scripts/bootstrap.py || echo "[WARN] Bootstrap deferred to first start"

# Shell form so Railway's $PORT env var is expanded at runtime
CMD ["/bin/sh", "-c", "gunicorn api.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120 --graceful-timeout 30"]
