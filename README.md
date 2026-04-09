# MarketForge AI

**UK AI Job Market Intelligence Platform**

Autonomous multi-department agentic AI system that continuously monitors, analyses, and distils the UK AI/ML job market into actionable intelligence. Nine specialised departments — each a LangGraph DeepAgent hierarchy — run on a twice-weekly schedule and produce skill demand rankings, salary benchmarks, sponsorship rates, career gap analysis, and emerging research signals.

[![CI](https://github.com/viraj97-sl/marketforge-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/viraj97-sl/marketforge-ai/actions)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.x-green)
![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-blue)

---

## What It Does

| Capability | Detail |
|---|---|
| **Job ingestion** | Scrapes Adzuna, Reed, Wellfound, specialist boards — ~200–800 roles/run |
| **NLP extraction** | 3-gate pipeline: taxonomy exact match → spaCy NER → Gemini LLM fallback |
| **Market analysis** | Skill demand index, salary percentiles, sponsorship rates, city distribution |
| **Career advisor** | Enter skills manually → AI gap analysis benchmarked against live data |
| **CV analyser** | Upload PDF/DOCX → instant ATS score (0–100, A+→D), skill gap plan (short/mid/long-term), GDPR-compliant |
| **Research signals** | arXiv + tech blogs monitored; predicts emerging skills 4–8 weeks early |
| **Weekly report** | Auto-generated LinkedIn-quality market briefing |
| **Dashboard** | 7-page Streamlit UI — Market Overview, Skill Intelligence, Salary & Geography, Career Advisor, Research Signals, Job Board, Pipeline Status |
| **REST API** | FastAPI with rate limiting, PII scrubbing, prompt injection guardrails |

---

## Architecture

Nine departments, each a LangGraph DeepAgent hierarchy running a Plan → Execute → Reflect → Output lifecycle. All state persists in the `market` schema in PostgreSQL.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Apache Airflow (scheduler)                  │
│                                                                 │
│  dag_ingest_primary ─► dag_weekly_analysis ─► dag_model_retrain│
└──────────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼───────────────────────────────────────────────┐
   │                    LangGraph DAG                          │
   │                                                           │
   │  [1] Data Collection  →  [2] ML Engineering               │
   │          ↓                       ↓                        │
   │  [3] Market Analysis  →  [4] Research Intelligence        │
   │          ↓                       ↓                        │
   │  [5] Content Studio   ←  [6] User Insights                │
   │                                                           │
   │  [7] QA & Testing   [8] Security   [9] Ops & Observability│
   └───────────────────────────────────────────────────────────┘
               │
   ┌───────────▼───────────────┐    ┌────────────────────────┐
   │  PostgreSQL (market.*)    │    │  Redis (dashboard cache)│
   │  • jobs                   │    └────────────────────────┘
   │  • job_skills             │    ┌────────────────────────┐
   │  • weekly_snapshots       │    │  ChromaDB (embeddings) │
   │  • research_signals       │    └────────────────────────┘
   │  • pipeline_runs          │    ┌────────────────────────┐
   │  • agent_state            │    │  MLflow (model registry│
   └───────────────────────────┘    └────────────────────────┘
```

| # | Department | Lead Agent | Responsibility |
|---|---|---|---|
| 1 | Data Collection | `DataCollectionLeadAgent` | Ingest 15+ UK job sources via connectors |
| 2 | ML Engineering | `MLEngineerLeadAgent` | Feature engineering, skill extraction model, salary prediction, hiring velocity forecast |
| 3 | Market Analysis | `MarketAnalystLeadAgent` | Skill demand trends, salary intelligence, sponsorship tracking, geo distribution |
| 4 | Research Intelligence | `ResearchLeadAgent` | arXiv monitoring, emerging tech signal detection |
| 5 | Content Studio | `ContentLeadAgent` | Weekly LinkedIn-quality market report |
| 6 | User Insights | `UserInsightsLeadAgent` | Personalised career gap analysis via API |
| 7 | QA & Testing | `QALeadAgent` | Data integrity, LLM output validation, model drift detection |
| 8 | Security | `SecurityLeadAgent` | Input sanitisation, PII scrubbing, prompt injection defence |
| 9 | Ops & Observability | `OpsLeadAgent` | Cost tracking, pipeline health, alert dispatch |

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Agent orchestration | LangGraph | 0.2.x |
| Pipeline scheduling | Apache Airflow | 2.9.x |
| LLM | Google Gemini Flash 2.0 + Pro | — |
| LLM observability | LangSmith | — |
| ML tracking | MLflow | 2.x |
| NLP (gate 2) | spaCy + en_core_web_sm | 3.8.x |
| Embeddings / dedup | sentence-transformers MiniLM | 3.x |
| Taxonomy matching | flashtext | 2.7 |
| Primary database | PostgreSQL + pgvector | 16 |
| Vector store | ChromaDB | — |
| Cache | Redis | 7.x |
| REST API | FastAPI + uvicorn | — |
| Dashboard | Streamlit | — |
| Containerisation | Docker Compose | — |
| Metrics | Prometheus | 2.52 |

---

## Project Structure

```
marketforge-ai/
├── api/
│   ├── main.py              # FastAPI app — 8 endpoints incl. CV analysis
│   └── security.py          # Rate limiting, auth middleware
├── airflow/
│   └── dags/                # 5 Airflow DAG definitions
├── dashboard/
│   └── app.py               # 7-page Streamlit dashboard
├── scripts/
│   ├── bootstrap.py         # DB init + taxonomy seed
│   ├── run_pipeline.py      # Manual one-shot pipeline runner
│   └── init_schemas.sql     # PostgreSQL DDL bootstrap
├── src/marketforge/
│   ├── agents/              # 9 departments, each with sub-agents + lead
│   │   ├── data_collection/
│   │   ├── ml_engineering/
│   │   ├── market_analysis/
│   │   ├── research_intelligence/
│   │   ├── content_studio/
│   │   ├── user_insights/
│   │   ├── qa/
│   │   ├── security/
│   │   └── ops/
│   ├── memory/
│   │   ├── postgres.py      # JobStore, SnapshotStore, PipelineRunStore
│   │   └── redis_cache.py   # DashboardCache with TTL + invalidation
│   ├── models/
│   │   ├── job.py           # RawJob, EnrichedJob, MarketSnapshot, …
│   │   └── state.py         # LangGraph state TypedDicts
│   ├── cv/                  # CV analysis module (GDPR-compliant, in-memory only)
│   │   ├── scanner.py       # Security gate: magic bytes, PDF JS, DOCX macros, ClamAV
│   │   ├── parser.py        # PDF (pdfplumber → pypdf fallback) + DOCX extraction
│   │   ├── ats_scorer.py    # 5-dimension ATS scoring: keywords/structure/readability/completeness/format
│   │   ├── gdpr.py          # PII scrubbing, consent gate, anonymous session token
│   │   └── gap_analyser.py  # ML gap prioritisation — demand×salary×recency scoring
│   ├── nlp/
│   │   ├── extractor.py     # NLP pipeline coordinator
│   │   └── taxonomy.py      # Gate1 (flashtext), Gate2 (spaCy), Gate3 (Gemini)
│   └── config.py            # Settings (Pydantic BaseSettings)
├── docker-compose.yml       # Full local stack
├── Dockerfile
├── pyproject.toml
└── .env                     # API keys + DB URLs (never commit)
```

---

## Quick Start

### Prerequisites

- Python 3.11
- Docker Desktop (for PostgreSQL, Redis, Airflow)
- Google Gemini API key (free tier sufficient)
- Adzuna API key (free — [register here](https://developer.adzuna.com/))
- Reed API key (free — [register here](https://www.reed.co.uk/developers/jobseeker))

### 1 — Clone and create virtual environment

```bash
git clone https://github.com/viraj97-sl/marketforge-ai.git
cd marketforge-ai
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2 — Install NLP models

```bash
python -m spacy download en_core_web_sm
pip install sentence-transformers langchain-google-genai
```

### 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` — minimum required keys:

```env
# Database
DATABASE_URL=postgresql+asyncpg://marketforge:marketforge@localhost:5432/marketforge
DATABASE_URL_SYNC=postgresql+psycopg2://marketforge:marketforge@localhost:5432/marketforge

# Redis
REDIS_URL=redis://localhost:6379/0

# LLMs
GOOGLE_API_KEY=your_gemini_api_key

# Job board connectors
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
REED_API_KEY=your_reed_api_key

# Optional (LangSmith tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=marketforge-ai
```

### 4 — Start infrastructure

```bash
docker-compose up -d postgres redis
```

Wait ~10 seconds for PostgreSQL to initialise, then:

```bash
python scripts/bootstrap.py
```

This creates all tables, seeds the skill taxonomy, and verifies connectivity.

### 5 — Run the first pipeline

```bash
python scripts/run_pipeline.py
```

Expect ~2–5 minutes. Typical output:
```
[pipeline] Ingested 312 raw jobs
[nlp] gate1=287 gate2=14 gate3=4 total_skills=1842
[analysis] Snapshot written — week 2026-04-07
[cache] Redis invalidated
```

### 6 — Start the API

```bash
# Windows
$env:PYTHONPATH="src"; uvicorn api.main:app --reload

# macOS / Linux
PYTHONPATH=src uvicorn api.main:app --reload
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### 7 — Start the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard at `http://localhost:8501`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System status, data freshness, job count |
| `GET` | `/api/v1/market/snapshot` | Latest weekly snapshot (skills, salary, sponsorship) |
| `GET` | `/api/v1/market/skills` | Skill demand by role category |
| `GET` | `/api/v1/market/trending` | Rising / declining skills (week-on-week) |
| `POST` | `/api/v1/career/analyse` | Career gap analysis for a given skill set |
| `POST` | `/api/v1/career/cv-analyse` | Upload CV (PDF/DOCX) → ATS score + gap plan (GDPR-compliant, no data stored) |
| `GET` | `/api/v1/jobs` | Browse indexed jobs with filters |
| `GET` | `/api/v1/pipeline/runs` | Recent pipeline execution history |

---

## Pipeline Schedule

Airflow runs automatically when `docker-compose up -d` includes the Airflow services:

```
Tuesday  07:00 UTC  — dag_ingest_primary     (Mon/Tue peak volume)
Thursday 07:00 UTC  — dag_ingest_primary     (Wed/Thu volume)
Monday   07:00 UTC  — dag_weekly_analysis    (weekly report + snapshot)
Sunday   02:00 UTC  — dag_model_retrain      (off-peak ML cycle)
Every 6h            — dag_dashboard_refresh  (cache refresh)
```

For one-off runs use `python scripts/run_pipeline.py`.

---

## Dashboard Pages

| Page | What you see |
|---|---|
| **Market Overview** | Top 20 skills bar chart, rising/declining skill signals, KPI strip |
| **Skill Intelligence** | Demand index table, treemap visualisation, week-on-week trend signals |
| **Salary & Geography** | Salary box plot (P25/P50/P75), sponsorship gauge, city bar chart |
| **Career Advisor** | Enter your skills → AI gap analysis, 90-day action plan, sector fit |
| **Research Signals** | Emerging techniques from arXiv + blogs before they appear in job ads |
| **Job Board** | Browse all indexed roles with filters; links to original job postings |
| **Pipeline Status** | Run history, LLM cost tracker, per-agent health monitor |

---

## NLP Extraction Pipeline

Three-gate skill extraction runs on every ingested job description:

```
Description text
      │
      ▼
 Gate 1 — flashtext taxonomy match (fast, zero-cost)
      │  ~85–90% of extractions
      ▼
 Gate 2 — spaCy NER (en_core_web_sm) — catches novel entities
      │  ~8–12% of extractions
      ▼
 Gate 3 — Gemini Flash LLM fallback (highest recall, ~$0.002/job)
           ~2–5% of extractions
```

---

## Cost Model

| Item | Cost |
|---|---|
| Gemini Flash (gate 3, ~50 jobs/run) | ~$0.02/run |
| Gemini Pro (career analysis, on-demand) | ~$0.01/query |
| PostgreSQL (Railway free tier) | $0/month |
| Redis (Railway free tier) | $0/month |
| **Total at 2 runs/week** | **~$0.20–0.40/month** |

Well within the $5/month infrastructure budget.

---

## Local Services

| Service | URL | Credentials |
|---|---|---|
| Dashboard | http://localhost:8501 | — |
| FastAPI | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5001 | — |
| Prometheus | http://localhost:9090 | — |
| PostgreSQL | localhost:5432 | marketforge / marketforge |
| Redis | localhost:6379 | — |

---

## Build Phases

- **Phase 1** — Data Foundation: PostgreSQL schema, skill taxonomy, 4 connectors, NLP gates 1+2
- **Phase 2** — Analysis Core: Market Analysis dept, Redis cache, MLflow, basic dashboard
- **Phase 3** — Agentic Intelligence: ML Engineering, Research Intelligence, Content Studio, LangSmith tracing
- **Phase 4** — User Features: Career Advisor API, Security dept, FastAPI hardening
- **Phase 5** — CV Analyser: ATS scoring, GDPR-compliant CV upload, ML gap prioritisation, Next.js frontend integration
- **Phase 6** *(ongoing)* — Ops dept, remaining connectors, Grafana, public launch

---

## Author

Viraj Bulugahapitiya · MSc Data Science (University of Hertfordshire) · 2026

Portfolio project demonstrating production-grade AI engineering: multi-agent orchestration, NLP pipelines, async FastAPI, PostgreSQL, Redis, Docker, MLflow, LangSmith, and Streamlit — all on a sub-$5/month infrastructure budget.

---

*Specification: [MarketForge_AI_Software_Spec_v1.0](docs/MarketForge_AI_Software_Spec_v1_0.docx)*
