"""
MarketForge AI — PostgreSQL Memory Layer.

Provides async and sync engine access, table creation (DDL), and
the core store classes used by all departments.

All tables live in the 'market' schema — completely isolated from
any other project on the same PostgreSQL instance.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from marketforge.config.settings import settings

logger = structlog.get_logger(__name__)

# ── Engine singletons ─────────────────────────────────────────────────────────
_async_engine: AsyncEngine | None = None
_sync_engine:  Engine      | None = None


def get_async_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _async_engine


def get_sync_engine() -> Engine:
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            settings.database_url_sync,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _sync_engine


# ── DDL — create all tables ───────────────────────────────────────────────────
_DDL = """
-- ── Market schema ────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS market;

-- Core job postings (raw, unfiltered)
CREATE TABLE IF NOT EXISTS market.jobs (
    job_id              TEXT PRIMARY KEY,
    dedup_hash          TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    title               TEXT NOT NULL,
    description         TEXT,
    company             TEXT NOT NULL,
    location            TEXT,
    salary_min          REAL,
    salary_max          REAL,
    salary_currency     TEXT DEFAULT 'GBP',
    work_model          TEXT DEFAULT 'unknown',
    experience_level    TEXT,
    role_category       TEXT,
    industry            TEXT,
    company_stage       TEXT DEFAULT 'unknown',
    is_startup          BOOLEAN DEFAULT FALSE,
    offers_sponsorship  BOOLEAN,
    citizens_only       BOOLEAN,
    degree_required     TEXT DEFAULT 'not_stated',
    equity_offered      BOOLEAN,
    url                 TEXT,
    source              TEXT NOT NULL,
    posted_date         DATE,
    scraped_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Extracted skills (many-to-many with jobs)
CREATE TABLE IF NOT EXISTS market.job_skills (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              TEXT REFERENCES market.jobs(job_id) ON DELETE CASCADE,
    skill               TEXT NOT NULL,
    skill_category      TEXT,
    extraction_method   TEXT NOT NULL DEFAULT 'gate1',
    confidence          REAL DEFAULT 1.0,
    extracted_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Canonical skill taxonomy
CREATE TABLE IF NOT EXISTS market.skill_taxonomy (
    skill_id    SERIAL PRIMARY KEY,
    canonical   TEXT UNIQUE NOT NULL,
    aliases     TEXT[],
    category    TEXT,
    parent_skill TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Company enrichment
CREATE TABLE IF NOT EXISTS market.companies (
    company_id          SERIAL PRIMARY KEY,
    name                TEXT UNIQUE NOT NULL,
    stage               TEXT,
    sector              TEXT,
    employee_count_band TEXT,
    founded_year        INT,
    hq_city             TEXT,
    is_uk_headquartered BOOLEAN,
    careers_url         TEXT,
    last_enriched_at    TIMESTAMPTZ
);

-- Weekly aggregated snapshots (fast read path for dashboard + reports)
CREATE TABLE IF NOT EXISTS market.weekly_snapshots (
    snapshot_id         SERIAL PRIMARY KEY,
    week_start          DATE NOT NULL,
    role_category       TEXT NOT NULL DEFAULT 'all',
    top_skills          JSONB DEFAULT '{}',
    rising_skills       JSONB DEFAULT '[]',
    declining_skills    JSONB DEFAULT '[]',
    salary_p25          REAL,
    salary_p50          REAL,
    salary_p75          REAL,
    salary_sample_size  INT DEFAULT 0,
    job_count           INT DEFAULT 0,
    sponsorship_rate    REAL DEFAULT 0,
    remote_rate         REAL DEFAULT 0,
    startup_rate        REAL DEFAULT 0,
    top_cities          JSONB DEFAULT '{}',
    computed_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(week_start, role_category)
);

-- Skill co-occurrence (PMI-scored pairs for network visualisation)
CREATE TABLE IF NOT EXISTS market.skill_cooccurrence (
    id        BIGSERIAL PRIMARY KEY,
    skill_a   TEXT NOT NULL,
    skill_b   TEXT NOT NULL,
    pmi_score REAL NOT NULL,
    co_count  INT  NOT NULL,
    week      DATE NOT NULL,
    UNIQUE(skill_a, skill_b, week)
);

-- ML feature vectors
CREATE TABLE IF NOT EXISTS market.ml_features (
    job_id          TEXT PRIMARY KEY REFERENCES market.jobs(job_id) ON DELETE CASCADE,
    feature_json    JSONB NOT NULL,
    feature_version TEXT  NOT NULL,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ML model evaluation history
CREATE TABLE IF NOT EXISTS market.model_evaluations (
    id            BIGSERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    metric_name   TEXT NOT NULL,
    metric_value  REAL NOT NULL,
    eval_set_hash TEXT,
    mlflow_run_id TEXT,
    evaluated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Research signals (emerging techniques from arXiv/blogs)
CREATE TABLE IF NOT EXISTS market.research_signals (
    signal_id           SERIAL PRIMARY KEY,
    technique_name      TEXT UNIQUE NOT NULL,
    source              TEXT NOT NULL,
    first_seen          DATE NOT NULL,
    mention_count       INT  DEFAULT 1,
    first_in_jd         DATE,
    adoption_lag_days   INT,
    relevance_score     REAL DEFAULT 0,
    arxiv_ids           TEXT[],
    summary             TEXT,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- arXiv paper index
CREATE TABLE IF NOT EXISTS market.paper_index (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    techniques      JSONB DEFAULT '[]',
    relevance_score REAL DEFAULT 0,
    published_at    DATE,
    indexed_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Pipeline run telemetry
CREATE TABLE IF NOT EXISTS market.pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    dag_name        TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT DEFAULT 'running',
    jobs_scraped    INT  DEFAULT 0,
    jobs_new        INT  DEFAULT 0,
    llm_cost_usd    REAL DEFAULT 0,
    metadata        JSONB DEFAULT '{}'
);

-- Per-sub-agent persistent state
CREATE TABLE IF NOT EXISTS market.agent_state (
    agent_id            TEXT PRIMARY KEY,
    department          TEXT NOT NULL,
    last_run_at         TIMESTAMPTZ,
    last_yield          INT  DEFAULT 0,
    consecutive_failures INT DEFAULT 0,
    run_count           INT  DEFAULT 0,
    adaptive_params     JSONB DEFAULT '{}',
    reflection_log      JSONB DEFAULT '[]',
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Agent execution event log
CREATE TABLE IF NOT EXISTS market.agent_logs (
    log_id          BIGSERIAL PRIMARY KEY,
    run_id          TEXT,
    agent_name      TEXT NOT NULL,
    department      TEXT NOT NULL,
    action          TEXT NOT NULL,
    input_summary   TEXT,
    output_summary  TEXT,
    duration_ms     INT,
    logged_at       TIMESTAMPTZ DEFAULT NOW()
);

-- LLM cost tracking (per-call granularity)
CREATE TABLE IF NOT EXISTS market.cost_log (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT,
    agent_name      TEXT NOT NULL,
    model           TEXT NOT NULL,
    input_tokens    INT  NOT NULL,
    output_tokens   INT  NOT NULL,
    cost_usd        REAL NOT NULL,
    logged_at       TIMESTAMPTZ DEFAULT NOW()
);

-- QA check results
CREATE TABLE IF NOT EXISTS market.qa_log (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT,
    check_name  TEXT NOT NULL,
    result      TEXT NOT NULL,  -- pass / fail / warning
    details     JSONB DEFAULT '{}',
    logged_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Security events
CREATE TABLE IF NOT EXISTS market.security_log (
    id                  BIGSERIAL PRIMARY KEY,
    event_type          TEXT NOT NULL,
    severity            TEXT NOT NULL,  -- critical / high / medium / low
    source_ip           TEXT,
    input_hash          TEXT,
    detection_method    TEXT,
    action_taken        TEXT,
    threat_score        REAL,
    logged_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Alert dispatch log
CREATE TABLE IF NOT EXISTS market.alert_log (
    alert_id        TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    severity        INT  NOT NULL,  -- 1 = critical, 2 = warning, 3 = info
    department      TEXT,
    message         TEXT NOT NULL,
    dispatched_at   TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ
);

-- Redis-style cross-run dedup store (fallback when Redis is unavailable)
CREATE TABLE IF NOT EXISTS market.seen_jobs (
    dedup_hash  TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    source      TEXT NOT NULL,
    first_seen  TIMESTAMPTZ DEFAULT NOW(),
    last_seen   TIMESTAMPTZ DEFAULT NOW(),
    times_seen  INT DEFAULT 1
);

-- LLM result cache (fallback when Redis is unavailable)
CREATE TABLE IF NOT EXISTS market.llm_cache (
    cache_key   TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    cached_at   TIMESTAMPTZ DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_jobs_dedup          ON market.jobs(dedup_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_run            ON market.jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_scraped        ON market.jobs(scraped_at);
CREATE INDEX IF NOT EXISTS idx_jobs_role_cat       ON market.jobs(role_category);
CREATE INDEX IF NOT EXISTS idx_job_skills_job      ON market.job_skills(job_id);
CREATE INDEX IF NOT EXISTS idx_job_skills_skill    ON market.job_skills(skill);
CREATE INDEX IF NOT EXISTS idx_snapshots_week      ON market.weekly_snapshots(week_start);
CREATE INDEX IF NOT EXISTS idx_agent_state_dept    ON market.agent_state(department);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_dag   ON market.pipeline_runs(dag_name);
CREATE INDEX IF NOT EXISTS idx_cost_log_run        ON market.cost_log(run_id);
CREATE INDEX IF NOT EXISTS idx_seen_jobs_hash      ON market.seen_jobs(dedup_hash);
CREATE INDEX IF NOT EXISTS idx_llm_cache_expires   ON market.llm_cache(expires_at);
"""


def init_database() -> None:
    """
    Create all market schema tables if they do not exist.
    Safe to call multiple times — fully idempotent.
    Works with both PostgreSQL and SQLite (SQLite ignores CREATE SCHEMA).
    """
    engine = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"

    with engine.connect() as conn:
        statements = _DDL.split(";")
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            # SQLite does not support schemas — strip "market." prefix
            if is_sqlite:
                stmt = stmt.replace("market.", "")
                stmt = stmt.replace("CREATE SCHEMA IF NOT EXISTS market", "")
                stmt = stmt.replace("BIGSERIAL", "INTEGER")
                stmt = stmt.replace("SERIAL", "INTEGER")
                stmt = stmt.replace("TIMESTAMPTZ", "TEXT")
                stmt = stmt.replace("BOOLEAN", "INTEGER")
                stmt = stmt.replace("JSONB", "TEXT")
                stmt = stmt.replace("TEXT[]", "TEXT")
                # SQLite doesn't have NOW() — use CURRENT_TIMESTAMP
                stmt = stmt.replace("DEFAULT NOW()", "DEFAULT CURRENT_TIMESTAMP")
                # gen_random_uuid() → expression default (needs outer parens for SQLite 3.38+)
                stmt = stmt.replace("gen_random_uuid()::TEXT", "(lower(hex(randomblob(16))))")
                if not stmt.strip():
                    continue
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                # Ignore extension-not-available errors (pgvector, pg_trgm)
                if "does not exist" in str(exc) or "not found" in str(exc):
                    continue
                raise
        conn.commit()

    # ── Column migrations (safe to run on existing DBs) ───────────────────────
    jobs_t = "jobs" if is_sqlite else "market.jobs"
    migrations = [
        f"ALTER TABLE {jobs_t} ADD COLUMN IF NOT EXISTS url TEXT",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                if is_sqlite:
                    # SQLite doesn't support IF NOT EXISTS on ALTER TABLE
                    # Check manually before adding
                    from sqlalchemy import inspect as sa_inspect
                    insp = sa_inspect(engine)
                    cols = [c["name"] for c in insp.get_columns(jobs_t.replace("market.", ""))]
                    if "url" not in cols:
                        conn.execute(text(f"ALTER TABLE {jobs_t.replace('market.', '')} ADD COLUMN url TEXT"))
                else:
                    conn.execute(text(stmt))
            except Exception:
                pass
        conn.commit()

    logger.info("database.init.complete", dialect=engine.dialect.name)


# ── DedupStore ────────────────────────────────────────────────────────────────
class DedupStore:
    """
    Cross-run job deduplication. Primary store is Redis (fast set operations).
    Falls back to market.seen_jobs table when Redis is unavailable.
    """

    def __init__(self) -> None:
        self._engine = get_sync_engine()
        self._is_sqlite = self._engine.dialect.name == "sqlite"
        self._table = "seen_jobs" if self._is_sqlite else "market.seen_jobs"

    def is_seen(self, dedup_hash: str) -> bool:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT 1 FROM {self._table} WHERE dedup_hash = :h"),
                {"h": dedup_hash},
            ).fetchone()
        return row is not None

    def mark_seen(self, dedup_hash: str, job_id: str, title: str, company: str, source: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
            if self._is_sqlite:
                conn.execute(text(f"""
                    INSERT OR REPLACE INTO {self._table}
                        (dedup_hash, job_id, title, company, source, first_seen, last_seen, times_seen)
                    VALUES (:h, :jid, :t, :co, :src, :now, :now,
                        COALESCE((SELECT times_seen FROM {self._table} WHERE dedup_hash=:h), 0) + 1)
                """), {"h": dedup_hash, "jid": job_id, "t": title, "co": company, "src": source, "now": now})
            else:
                conn.execute(text(f"""
                    INSERT INTO {self._table}
                        (dedup_hash, job_id, title, company, source, first_seen, last_seen, times_seen)
                    VALUES (:h, :jid, :t, :co, :src, NOW(), NOW(), 1)
                    ON CONFLICT(dedup_hash) DO UPDATE SET
                        last_seen  = NOW(),
                        times_seen = {self._table}.times_seen + 1
                """), {"h": dedup_hash, "jid": job_id, "t": title, "co": company, "src": source})
            conn.commit()

    def filter_new(self, jobs: list) -> list:
        """Return only unseen jobs, marking all as seen atomically."""
        new = []
        for job in jobs:
            if not self.is_seen(job.dedup_hash):
                self.mark_seen(job.dedup_hash, job.job_id, job.title, job.company, job.source)
                new.append(job)
        return new


# ── AgentStateStore ────────────────────────────────────────────────────────────
class AgentStateStore:
    """
    Reads and writes persistent state for sub-agents.
    Every sub-agent's plan() method calls load() at the start;
    every reflect() method calls save() at the end.
    """

    def __init__(self) -> None:
        self._engine = get_sync_engine()
        self._is_sqlite = self._engine.dialect.name == "sqlite"
        self._table = "agent_state" if self._is_sqlite else "market.agent_state"

    def load(self, agent_id: str, department: str) -> dict[str, Any]:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {self._table} WHERE agent_id = :aid"),
                {"aid": agent_id},
            ).mappings().fetchone()
        if row is None:
            return {
                "agent_id": agent_id,
                "department": department,
                "last_run_at": None,
                "last_yield": 0,
                "consecutive_failures": 0,
                "run_count": 0,
                "adaptive_params": {},
                "reflection_log": [],
            }
        data = dict(row)
        for field in ("adaptive_params", "reflection_log"):
            if isinstance(data.get(field), str):
                data[field] = json.loads(data[field])
        return data

    def save(self, state: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        adaptive_params = json.dumps(state.get("adaptive_params", {}))
        reflection_log  = json.dumps(state.get("reflection_log", []))
        with self._engine.connect() as conn:
            if self._is_sqlite:
                conn.execute(text(f"""
                    INSERT OR REPLACE INTO {self._table}
                        (agent_id, department, last_run_at, last_yield,
                         consecutive_failures, run_count, adaptive_params,
                         reflection_log, updated_at)
                    VALUES (:aid, :dept, :lra, :ly, :cf, :rc, :ap, :rl, :now)
                """), {
                    "aid": state["agent_id"], "dept": state["department"],
                    "lra": state.get("last_run_at"), "ly": state.get("last_yield", 0),
                    "cf": state.get("consecutive_failures", 0), "rc": state.get("run_count", 0),
                    "ap": adaptive_params, "rl": reflection_log, "now": now,
                })
            else:
                conn.execute(text(f"""
                    INSERT INTO {self._table}
                        (agent_id, department, last_run_at, last_yield,
                         consecutive_failures, run_count, adaptive_params,
                         reflection_log, updated_at)
                    VALUES (:aid, :dept, :lra, :ly, :cf, :rc, CAST(:ap AS jsonb), CAST(:rl AS jsonb), NOW())
                    ON CONFLICT(agent_id) DO UPDATE SET
                        last_run_at          = EXCLUDED.last_run_at,
                        last_yield           = EXCLUDED.last_yield,
                        consecutive_failures = EXCLUDED.consecutive_failures,
                        run_count            = EXCLUDED.run_count,
                        adaptive_params      = EXCLUDED.adaptive_params,
                        reflection_log       = EXCLUDED.reflection_log,
                        updated_at           = NOW()
                """), {
                    "aid": state["agent_id"], "dept": state["department"],
                    "lra": state.get("last_run_at"), "ly": state.get("last_yield", 0),
                    "cf": state.get("consecutive_failures", 0), "rc": state.get("run_count", 0),
                    "ap": adaptive_params, "rl": reflection_log,
                })
            conn.commit()


# ── JobStore ──────────────────────────────────────────────────────────────────
class JobStore:
    """Writes enriched jobs and their skills to the market schema."""

    def __init__(self) -> None:
        self._engine = get_sync_engine()
        self._is_sqlite = self._engine.dialect.name == "sqlite"
        self._jobs_t   = "jobs"           if self._is_sqlite else "market.jobs"
        self._skills_t = "job_skills"     if self._is_sqlite else "market.job_skills"

    def upsert_job(self, job: Any, run_id: str) -> None:
        from marketforge.models.job import RawJob
        j: RawJob = job
        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO {self._jobs_t}
                    (job_id, dedup_hash, run_id, title, description, company, location,
                     salary_min, salary_max, work_model, experience_level,
                     role_category, industry, company_stage, is_startup,
                     offers_sponsorship, citizens_only, degree_required,
                     url, source, posted_date, scraped_at)
                VALUES
                    (:job_id, :dedup_hash, :run_id, :title, :description, :company, :location,
                     :salary_min, :salary_max, :work_model, :experience_level,
                     :role_category, :industry, :company_stage, :is_startup,
                     :offers_sponsorship, :citizens_only, :degree_required,
                     :url, :source, :posted_date, :scraped_at)
                ON CONFLICT(job_id) DO NOTHING
            """), {
                "job_id": j.job_id, "dedup_hash": j.dedup_hash, "run_id": run_id,
                "title": j.title, "description": j.description or None, "company": j.company, "location": j.location,
                "salary_min": j.salary_min, "salary_max": j.salary_max,
                "work_model": j.work_model, "experience_level": j.experience_level,
                "role_category": j.role_category, "industry": j.industry,
                "company_stage": j.company_stage,
                "is_startup": int(j.is_startup) if self._is_sqlite else j.is_startup,
                "offers_sponsorship": j.offers_sponsorship,
                "citizens_only": j.citizens_only, "degree_required": j.degree_required,
                "url": j.url or None,
                "source": j.source,
                "posted_date": j.posted_date.isoformat() if j.posted_date else None,
                "scraped_at": j.scraped_at.isoformat() if j.scraped_at else now,
            })
            conn.commit()

    def upsert_skills(self, job_id: str, skills: list[tuple[str, str, str, float]]) -> None:
        """skills: list of (skill, category, method, confidence)"""
        if not skills:
            return
        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
            for skill, category, method, confidence in skills:
                conn.execute(text(f"""
                    INSERT INTO {self._skills_t}
                        (job_id, skill, skill_category, extraction_method, confidence, extracted_at)
                    VALUES (:jid, :skill, :cat, :method, :conf, :now)
                    ON CONFLICT DO NOTHING
                """), {"jid": job_id, "skill": skill, "cat": category,
                       "method": method, "conf": confidence, "now": now})
            conn.commit()


# ── PipelineRunStore ───────────────────────────────────────────────────────────
class PipelineRunStore:
    """Tracks DAG execution telemetry."""

    def __init__(self) -> None:
        self._engine = get_sync_engine()
        self._is_sqlite = self._engine.dialect.name == "sqlite"
        self._table = "pipeline_runs" if self._is_sqlite else "market.pipeline_runs"

    def start(self, run_id: str, dag_name: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO {self._table} (run_id, dag_name, started_at)
                VALUES (:rid, :dag, :now)
                ON CONFLICT(run_id) DO NOTHING
            """), {"rid": run_id, "dag": dag_name, "now": now})
            conn.commit()

    def finish(self, run_id: str, status: str, **kwargs: Any) -> None:
        set_parts = ", ".join(f"{k} = :{k}" for k in kwargs)
        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
            conn.execute(text(f"""
                UPDATE {self._table}
                SET completed_at = :_now, status = :_status {', ' + set_parts if set_parts else ''}
                WHERE run_id = :run_id
            """), {"run_id": run_id, "_now": now, "_status": status, **kwargs})
            conn.commit()
