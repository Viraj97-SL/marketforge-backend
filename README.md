# MarketForge Backend

**FastAPI + APScheduler Deployment Layer for MarketForge AI**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/deploy-Railway-blueviolet)](https://railway.app)
![Gunicorn](https://img.shields.io/badge/Gunicorn-2_workers-lightgrey)
![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-blue)
![Redis](https://img.shields.io/badge/Redis-7.x-red)

> Live API at the backend for **[marketforge.digital](https://marketforge.digital)**

---

## What Is This?

This repo is the **production deployment layer** for MarketForge AI. It hosts the FastAPI application and the APScheduler pipeline worker that runs on Railway. It contains no agent or ML code — all intelligence lives in the [`marketforge-ai`](https://github.com/Viraj97-SL/marketforge-ai) core package, installed as a git dependency at build time.

The separation of concerns is deliberate:
- **`marketforge-ai`** — all agent logic, LangGraph graphs, NLP pipelines, ML models, CV analyser
- **`marketforge-backend`** ← you are here — infrastructure: HTTP server, scheduling, rate limiting, CORS, deployment config

---

## Three-Repo Architecture

| Repo | Role | Deployed on |
|---|---|---|
| [`marketforge-ai`](https://github.com/Viraj97-SL/marketforge-ai) | Core package: 9 agents, LangGraph graphs, ML/NLP, CV analyser | Installed as git package |
| **`marketforge-backend`** ← you are here | FastAPI REST API + APScheduler worker | Railway |
| `marketforge-frontend` | Next.js 14 dashboard | Vercel |

---

## Repository Layout

```
marketforge-backend/
├── api/
│   ├── main.py          # FastAPI app — 10 endpoints, CORS, rate limiting, security middleware
│   └── security.py      # SecurityMiddleware: per-IP rate limiting, slow-request detection
├── worker.py            # APScheduler pipeline worker (long-lived Railway process)
├── src/marketforge/     # Synced copies of core files (kept in lock-step with marketforge-ai)
├── scripts/
│   ├── bootstrap.py     # DB schema init + 109-skill taxonomy seed
│   └── run_pipeline.py  # Manual one-shot pipeline trigger
├── tests/               # API integration tests
├── Dockerfile           # Multi-stage build; pre-installs heavy deps for layer caching
├── railway.toml         # Railway service config (api + worker services)
├── requirements.txt     # Heavy dependency pre-install (avoids cold rebuild every deploy)
└── pyproject.toml       # Slim manifest — core installed from git
```

---

## Core Dependency

All agent, graph, NLP, and ML code is installed directly from the `marketforge-ai` git repository:

```toml
# pyproject.toml
[project]
dependencies = [
  "marketforge-ai @ git+https://github.com/Viraj97-SL/marketforge-ai.git@main"
]
```

Railway's Dockerfile pulls this at build time, pinned to `main`. No agent code is duplicated.

---

## API Endpoints

All user-facing inputs pass through the LangGraph Department 8 security graph (`run_security_check()`) before any LLM call. All endpoints are rate-limited via Redis.

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | 100/min | System status, data freshness, job count |
| `GET` | `/api/v1/market/snapshot` | 100/min | Latest weekly snapshot: skills, salary, sponsorship |
| `GET` | `/api/v1/market/skills` | 100/min | Skill demand index by role category |
| `GET` | `/api/v1/market/salary` | 100/min | Salary p25/p50/p75 benchmarks |
| `GET` | `/api/v1/market/trending` | 100/min | Rising / declining skills week-on-week |
| `GET` | `/api/v1/jobs` | 100/min | Browse indexed roles (filter by role, work model, visa) |
| `POST` | `/api/v1/career/analyse` | 10/min | SBERT semantic match + Gemini 2.5 Pro career narrative |
| `POST` | `/api/v1/career/cv-analyse` | 3/hour | PDF/DOCX upload → ATS score + GDPR-compliant gap plan |
| `GET` | `/api/v1/pipeline/runs` | 100/min | Recent pipeline execution history |
| `GET` | `/metrics` | Bearer token | Prometheus metrics |

### CV Analyser — what happens on POST `/cv-analyse`

```
1. Rate limit check (3/IP/hour)
2. Input validation (injection defence via LangGraph security graph)
3. GDPR consent gate (403 if consent=false)
4. File security scan (magic bytes, PDF JS, AV signatures)
5. In-memory parse (PDF: pdfplumber → pypdf; DOCX: python-docx)
6. PII scrub (email, UK phone, NI number, postcode, DOB, address)
7. ATS scoring (5 dimensions, deterministic, no LLM)
8. ML gap analysis (demand × salary × recency scoring)
9. Gemini 2.5 Flash gap plan (seeded with ML-ranked buckets only)
10. Output guardrails → return CVAnalysisReport (data_retained=False)
```

No CV text, extracted data, or PII is written to the database at any step.

---

## Security Middleware

`SecurityMiddleware` in `api/security.py` runs on every request before routing:

- **Per-IP rate limiting** via Redis sliding window: 100 req/min (market data), 10 req/min (career), 3 req/hour (CV analysis)
- **Slow request detection**: logs `security.slow_request` warning for responses exceeding 10 seconds with IP, path, and duration
- **CORS**: `ALLOWED_ORIGINS` env var (defaults to `*` if unset)
- **Security headers**: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`
- **Input validation**: all user text fields pass through `validate_input()` from the LangGraph security graph before any LLM call — catches prompt injection, oversized payloads, and content policy violations

---

## Pipeline Worker

`worker.py` runs as a separate long-lived Railway service using APScheduler. It imports and invokes the LangGraph pipeline entry points on a fixed cron schedule — a lightweight replacement for Apache Airflow (which can't run efficiently on Railway's hobby tier).

### Schedule (UTC)

| Job | Cron | What runs |
|---|---|---|
| `ingest` | Tue + Thu 07:00 | `run_data_collection_pipeline()` → NLP extraction → `run_market_analysis_pipeline()` → Redis cache invalidation |
| `analysis` | Mon 07:00 | `run_market_analysis_pipeline()` only (weekly snapshot + email report, no scrape) |
| `retrain` | Sun 02:00 | `run_ml_pipeline()` — PSI drift check → retrain if drift exceeds threshold |
| `cache` | every 6h | Redis dashboard cache refresh |

### Manual trigger

```bash
# Trigger specific job immediately (exits after completion)
python worker.py --run-now ingest
python worker.py --run-now analysis
python worker.py --run-now retrain
python worker.py --run-now cache

# Or set env var in Railway to run on next deploy and auto-clear:
# RUN_NOW_ON_START=ingest
```

---

## Railway Deployment

Two Railway services are deployed from this single repo:

| Service | Start command | Notes |
|---|---|---|
| `api` | `gunicorn api.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT` | 2 workers; SBERT model pre-warmed at startup lifespan |
| `worker` | `python worker.py` | Long-lived process, APScheduler blocks |

Both services share the same PostgreSQL and Redis addons; Railway injects `DATABASE_URL` and `REDIS_URL` automatically.

### How the SBERT model is handled

The `sentence-transformers/all-MiniLM-L6-v2` model (~80 MB) is lazy-loaded as a process-level singleton. With 2 gunicorn workers, each worker loads its own copy on first use. To prevent the cold-start latency (25–35 seconds) from timing out browser connections, the model is now **pre-warmed at startup lifespan** — both workers have the model ready before the first request is served:

```python
@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(_get_sbert)   # warm both workers at startup
    logger.info("sbert.warmed")
    yield
```

### Environment variables (set in Railway dashboard)

| Variable | Source | Required |
|---|---|---|
| `DATABASE_URL` | Auto-injected by Railway PostgreSQL addon | Yes |
| `DATABASE_URL_SYNC` | Manually set: same host/db as DATABASE_URL but `postgresql+psycopg2://` prefix | Yes |
| `REDIS_URL` | Auto-injected by Railway Redis addon | Yes |
| `GEMINI_API_KEY` | Google AI Studio | Yes |
| `LANGCHAIN_API_KEY` | LangSmith | Yes |
| `LANGCHAIN_TRACING_V2` | `true` | Yes |
| `LANGCHAIN_PROJECT` | `marketforge-ai` | Yes |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | Adzuna developer portal | Yes |
| `REED_API_KEY` | Reed developer portal | Yes |
| `TAVILY_API_KEY` | Tavily | For research dept |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | Gmail app password | For weekly report |
| `REPORT_RECIPIENT_EMAIL` | Target email for weekly report | For weekly report |
| `ENVIRONMENT` | `production` | Yes |
| `LOG_LEVEL` | `INFO` | Yes |
| `LOG_FORMAT` | `json` | Yes |
| `ALLOWED_ORIGINS` | Comma-separated Vercel URLs | Recommended |

---

## Database Schema

All tables live in the `market` schema in PostgreSQL:

| Table | Purpose | Key constraint |
|---|---|---|
| `market.jobs` | Raw + enriched job postings | `ON CONFLICT(job_id) DO UPDATE SET scraped_at` |
| `market.job_skills` | Extracted skills per job | PK `(job_id, skill)` |
| `market.seen_jobs` | Cross-run dedup store | PK `dedup_hash`, TTL-gated |
| `market.weekly_snapshots` | Aggregated market stats | PK `(week_start, role_category)` |
| `market.pipeline_runs` | Telemetry per worker run | — |
| `market.agent_state` | Adaptive params per agent | — |

`dedup_hash` is SHA-256[:16] of `(title.lower(), company.lower(), location.lower())`.

Market analysis time window: `WHERE scraped_at >= week_start` where `week_start = date.today() - timedelta(days=date.today().weekday())` (Monday of current week). The `touch_scraped_at()` call in `DataCollectionLeadAgent` refreshes `scraped_at=NOW()` for ALL ~525 raw job IDs on every run — ensuring already-seen jobs stay within the analysis window.

---

## Local Development

```bash
git clone https://github.com/Viraj97-SL/marketforge-backend.git
cd marketforge-backend

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

Create `.env`:

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
ENVIRONMENT=development
```

Start local PostgreSQL + Redis:

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=marketforge -e POSTGRES_USER=marketforge -e POSTGRES_DB=marketforge postgres:16
docker run -d -p 6379:6379 redis:7
```

Bootstrap the database:

```bash
python scripts/bootstrap.py
```

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

Run a one-shot pipeline:

```bash
python worker.py --run-now ingest
```

---

## Observability

**LangSmith**: Set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` to automatically trace every `graph.ainvoke()` call to [studio.langsmith.com](https://studio.langsmith.com) → Projects → marketforge-ai. Every node's input and output is captured with timing.

**Prometheus**: `GET /metrics` (requires `METRICS_TOKEN` Bearer header in production). Exposes per-endpoint request count, latency histograms, and error rates.

**structlog**: All logs are structured JSON (`LOG_FORMAT=json` in production). Key log events:
- `api.startup` — service boot with dialect and version
- `sbert.warmed` — SBERT model pre-warm complete at startup
- `cv.endpoint.complete` — session token, ATS score, grade, ML gap count
- `security.slow_request` — IP, path, duration ms for requests > 10s
- `worker.ingest.start` / `worker.ingest.done` — pipeline run telemetry

---

## Author

**Viraj Bulugahapitiya** · AI Engineer · MSc Data Science, University of Hertfordshire (2026)

[marketforge.digital](https://marketforge.digital) · [GitHub](https://github.com/Viraj97-SL)
