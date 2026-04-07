# MarketForge AI Backend — Railway Production Dockerfile
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time (avoids cold-start delays)
RUN python -m spacy download en_core_web_sm || echo "[WARN] spaCy unavailable"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "[WARN] SBERT unavailable"

# Copy source
COPY . .
RUN pip install --no-cache-dir -e . || true

# Bootstrap DB schema (idempotent)
RUN python scripts/bootstrap.py || echo "[WARN] Bootstrap deferred to first start"

# Shell form so Railway's $PORT env var is expanded at runtime
CMD ["/bin/sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
