# MarketForge Backend

**FastAPI + APScheduler deployment layer for MarketForge AI**

This repo is the production backend service. It contains no agent code — all intelligence lives in the [`marketforge-ai`](https://github.com/Viraj97-SL/marketforge-ai) core package, which this service installs as a git dependency.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/deploy-Railway-blueviolet)](https://railway.app)

---

## Repository layout (3-repo architecture)

| Repo | Purpose |
|---|---|
| [`marketforge-ai`](https://github.com/Viraj97-SL/marketforge-ai) | Core package — all 9 LangGraph agents, ML/NLP pipelines |
| **`marketforge-backend`** ← you are here | FastAPI API + APScheduler worker — deployed to Railway |
| `marketforge-frontend` | Next.js frontend |

---

## What this repo contains

```
marketforge-backend/
├── api/
│   ├── main.py          # FastAPI app — 9 endpoints (career, market, CV, health, metrics)
│   └── security.py      # Rate limiting middleware + LangGraph security pipeline wiring
├── worker.py            # APScheduler pipeline worker (replaces Airflow for Railway)
├── scripts/             # DB bootstrap, migration helpers
├── tests/               # API + integration tests
├── Dockerfile           # Multi-service Railway build
├── railway.toml         # Railway service config (api + worker services)
├── requirements.txt     # Heavy dep pre-install for Docker layer caching
└── pyproject.toml       # Slim manifest — core installed from git
```

---

## Core dependency

All agent, graph, ML, and NLP code is installed from the core package:

```toml
# pyproject.toml
"marketforge-ai @ git+https://github.com/Viraj97-SL/marketforge-ai.git@main"
```

When Railway builds the Docker image, it pulls the core package at the pinned branch/commit. No agent code is duplicated here.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System status, data freshness, job count |
| `GET` | `/api/v1/market/snapshot` | Latest weekly snapshot (skills, salary, sponsorship) |
| `GET` | `/api/v1/market/skills` | Skill demand by role category |
| `GET` | `/api/v1/market/salary` | Salary benchmarks by role |
| `GET` | `/api/v1/market/trending` | Rising / declining skills week-on-week |
| `POST` | `/api/v1/career/analyse` | Personalised career gap analysis (SBERT + Gemini 2.5 Pro) |
| `POST` | `/api/v1/career/cv-analyse` | Upload PDF/DOCX → ATS score + skill gap plan (GDPR, no data stored) |
| `GET` | `/api/v1/jobs` | Browse indexed roles with filters |
| `GET` | `/metrics` | Prometheus metrics |

All user-facing inputs pass through the LangGraph Department 8 security graph (`run_security_check()`) before any LLM call.

---

## Pipeline schedule (APScheduler)

`worker.py` runs as a long-lived Railway service. It calls the LangGraph pipeline entry points on the same schedule the Airflow DAGs used:

```
Tuesday + Thursday  07:00 UTC  — run_data_collection_pipeline()
                                 run_market_analysis_pipeline()    (full ingest)
Monday              07:00 UTC  — run_market_analysis_pipeline()    (snapshot only)
Sunday              02:00 UTC  — run_ml_pipeline()                 (PSI drift → retrain)
Every 6h                       — Redis cache refresh
```

Trigger manually: `python worker.py --run-now ingest|analysis|retrain|cache`

---

## Railway deployment

Two services are deployed from this repo — configure both in the Railway dashboard:

| Service | Start command |
|---|---|
| `api` | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| `worker` | `python worker.py` |

Both services share the same PostgreSQL and Redis addons (Railway injects `DATABASE_URL` and `REDIS_URL` automatically).

**LangGraph checkpointing:** `AsyncPostgresSaver` automatically uses `DATABASE_URL` for graph state persistence. All 8 pipeline graphs checkpoint to PostgreSQL — node-level retry is possible without re-running the full pipeline.

**LangSmith tracing:** set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=marketforge-ai` in the Railway environment. Every pipeline run traces to [studio.langsmith.com](https://studio.langsmith.com) automatically.

### Environment variables (set in Railway dashboard)

| Variable | Source |
|---|---|
| `DATABASE_URL` | Auto-injected by Railway PostgreSQL addon |
| `REDIS_URL` | Auto-injected by Railway Redis addon |
| `GEMINI_API_KEY` | Google AI Studio |
| `LANGCHAIN_API_KEY` | LangSmith |
| `LANGCHAIN_TRACING_V2` | `true` |
| `LANGCHAIN_PROJECT` | `marketforge-ai` |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | Adzuna developer portal |
| `REED_API_KEY` | Reed developer portal |
| `TAVILY_API_KEY` | Tavily |
| `SMTP_*` | Gmail app password for weekly report dispatch |
| `ENVIRONMENT` | `production` |

---

## Local development

```bash
git clone https://github.com/Viraj97-SL/marketforge-backend.git
cd marketforge-backend
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                      # macOS / Linux

pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

Create `.env` (copy the vars from Railway dashboard or from the core repo's env):

```env
DATABASE_URL=postgresql+asyncpg://marketforge:marketforge@localhost:5432/marketforge
DATABASE_URL_SYNC=postgresql+psycopg2://marketforge:marketforge@localhost:5432/marketforge
REDIS_URL=redis://localhost:6379/0
GEMINI_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=marketforge-ai
ADZUNA_APP_ID=...
ADZUNA_APP_KEY=...
REED_API_KEY=...
```

Start local PostgreSQL + Redis:

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=marketforge postgres:16
docker run -d -p 6379:6379 redis:7
```

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Run the worker (one-shot):

```bash
python worker.py --run-now ingest
```

---

## Author

Viraj Bulugahapitiya · MSc Data Science (University of Hertfordshire) · 2026
