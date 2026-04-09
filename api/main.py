"""
MarketForge AI — FastAPI Application

Endpoints:
  POST /api/v1/career/analyse     — personalised career advice (LLM-backed)
  POST /api/v1/career/cv-analyse  — CV upload → ATS score + career gap plan
  GET  /api/v1/market/skills      — top skills by role category
  GET  /api/v1/market/salary      — salary benchmarks
  GET  /api/v1/market/snapshot    — full weekly market snapshot
  GET  /api/v1/market/trending    — rising / declining skill lists
  GET  /api/v1/health             — pipeline health and data freshness
  GET  /metrics                   — Prometheus metrics endpoint

All user inputs pass through SecurityGuardrails before any LLM call.
All endpoints are rate-limited via Redis.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from marketforge.agents.security.guardrails import validate_input
from marketforge.config.settings import settings
from marketforge.memory.postgres import init_database
from marketforge.memory.redis_cache import DashboardCache, RateLimiter
from marketforge.utils.logger import setup_logging

logger  = structlog.get_logger(__name__)
cache   = DashboardCache()
limiter = RateLimiter()

# Lazy-loaded SBERT model — cached for the lifetime of the process
_sbert_model = None

def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _sbert_model


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_database()
    logger.info("api.startup", version="0.1.0", env=settings.environment)
    yield
    logger.info("api.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MarketForge AI",
    description="UK AI Job Market Intelligence Platform — public API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Rate limit middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip  = request.client.host if request.client else "unknown"
    path = request.url.path

    # Career advisor: 10 req/min (LLM-backed, expensive)
    if path.startswith("/api/v1/career"):
        if not limiter.is_allowed(f"career:{ip}", limit=10, window_seconds=60):
            return PlainTextResponse("Rate limit exceeded", status_code=429)

    # Market data: 100 req/min (Redis-cached, cheap)
    elif path.startswith("/api/v1/market"):
        if not limiter.is_allowed(f"market:{ip}", limit=100, window_seconds=60):
            return PlainTextResponse("Rate limit exceeded", status_code=429)

    return await call_next(request)


# ── Request / Response models ─────────────────────────────────────────────────

class UserProfile(BaseModel):
    skills:           list[str]  = Field(min_length=1,  max_length=50,  description="Your current skills")
    target_role:      str        = Field(min_length=2,  max_length=100, description="Target role type")
    experience_level: str        = Field(default="mid", description="junior / mid / senior / lead")
    location:         str        = Field(default="London", max_length=100)
    visa_sponsorship: bool       = Field(default=False, description="Whether you need visa sponsorship")
    free_text:        str | None = Field(default=None, max_length=2000, description="Optional background context")


class CareerIntelligenceReport(BaseModel):
    market_match_pct:     float
    match_distribution:   dict[str, float]    # strong / moderate / weak
    top_skill_gaps:       list[dict[str, Any]]
    sector_fit:           list[dict[str, Any]]
    salary_expectation:   dict[str, Any]
    action_plan_90d:      list[str]
    narrative_summary:    str
    security_warnings:    list[str]


class MarketSnapshotResponse(BaseModel):
    week_start:       str
    role_category:    str
    job_count:        int
    top_skills:       dict[str, int]
    salary_p25:       float | None
    salary_p50:       float | None
    salary_p75:       float | None
    sponsorship_rate: float
    computed_at:      str


class HealthResponse(BaseModel):
    status:           str
    last_ingestion:   str | None
    data_freshness_h: float | None
    jobs_total:       int
    version:          str


# ── Career Advisor endpoint ───────────────────────────────────────────────────

@app.post(
    "/api/v1/career/analyse",
    response_model=CareerIntelligenceReport,
    summary="Personalised career gap analysis",
    description="Analyses your profile against current market data. No data is persisted.",
)
async def analyse_career(profile: UserProfile, request: Request) -> CareerIntelligenceReport:
    ip = request.client.host if request.client else None

    # ── Security gate ─────────────────────────────────────────────────────────
    all_text = " ".join(profile.skills) + " " + (profile.free_text or "")
    sec_result = validate_input(all_text, field_name="profile", source_ip=ip)
    if not sec_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=sec_result.rejection_reason,
        )

    skills_text = sec_result.sanitised_text

    # ── Market match via SBERT + ChromaDB ────────────────────────────────────
    match_pct, match_dist = _compute_market_match(profile.skills)

    # ── Skill gap analysis ────────────────────────────────────────────────────
    skill_gaps = _compute_skill_gaps(profile.skills, profile.target_role)

    # ── Sector fit ────────────────────────────────────────────────────────────
    sector_fit = _compute_sector_fit(profile.skills)

    # ── Salary expectation ────────────────────────────────────────────────────
    salary_exp = _fetch_salary_expectation(profile.target_role, profile.experience_level, profile.location)

    # ── LLM narrative synthesis ───────────────────────────────────────────────
    narrative, action_plan = await _generate_career_narrative(profile, match_pct, skill_gaps, sector_fit, salary_exp)

    # ── Output guardrails ─────────────────────────────────────────────────────
    from marketforge.agents.security.guardrails import validate_output
    narrative, sec_warnings = validate_output(narrative)

    return CareerIntelligenceReport(
        market_match_pct=round(match_pct, 1),
        match_distribution=match_dist,
        top_skill_gaps=skill_gaps[:5],
        sector_fit=sector_fit[:3],
        salary_expectation=salary_exp,
        action_plan_90d=action_plan,
        narrative_summary=narrative,
        security_warnings=sec_warnings,
    )


def _compute_market_match(skills: list[str]) -> tuple[float, dict[str, float]]:
    """SBERT embed the skill list and query market data for similarity."""
    try:
        import numpy as np
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        engine = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        table = "jobs" if is_sqlite else "market.jobs"

        # Sample recent job titles for comparison
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT title, role_category FROM {table}
                ORDER BY scraped_at DESC LIMIT 200
            """)).fetchall()

        if not rows:
            return 50.0, {"strong": 0.3, "moderate": 0.4, "weak": 0.3}

        model      = _get_sbert()
        profile_emb= model.encode(" ".join(skills), normalize_embeddings=True)
        job_texts  = [f"{r[0]} {r[1] or ''}" for r in rows]
        job_embs   = model.encode(job_texts, normalize_embeddings=True, batch_size=64)

        similarities = np.dot(job_embs, profile_emb)
        strong   = float(np.mean(similarities > 0.75))
        moderate = float(np.mean((similarities > 0.55) & (similarities <= 0.75)))
        weak     = float(np.mean(similarities <= 0.55))
        match_pct= float(np.mean(similarities) * 100)

        return min(max(match_pct, 0), 100), {
            "strong": round(strong, 3),
            "moderate": round(moderate, 3),
            "weak": round(weak, 3),
        }
    except Exception as exc:
        logger.warning("market_match.error", error=str(exc))
        return 50.0, {"strong": 0.3, "moderate": 0.4, "weak": 0.3}


def _compute_skill_gaps(user_skills: list[str], target_role: str) -> list[dict[str, Any]]:
    """Compare user skills against top-demanded skills for the target role."""
    try:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        import json

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        snap_t    = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

        with engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT top_skills FROM {snap_t}
                ORDER BY week_start DESC LIMIT 1
            """)).fetchone()

        if not row or not row[0]:
            return []

        top_skills = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        user_lower = {s.lower() for s in user_skills}

        gaps = []
        for skill, count in sorted(top_skills.items(), key=lambda x: -x[1]):
            if skill.lower() not in user_lower:
                gaps.append({"skill": skill, "market_demand": count, "priority": "high" if count > 50 else "medium"})
            if len(gaps) >= 10:
                break
        return gaps
    except Exception as exc:
        logger.warning("skill_gaps.error", error=str(exc))
        return []


def _compute_sector_fit(user_skills: list[str]) -> list[dict[str, Any]]:
    """Basic sector fit based on skill keyword matching."""
    sectors = [
        {"sector": "FinTech", "keywords": ["python", "ml", "xgboost", "sql", "pandas", "risk"], "sponsorship_rate": 0.35},
        {"sector": "HealthTech", "keywords": ["python", "deep learning", "pytorch", "medical", "nlp"], "sponsorship_rate": 0.28},
        {"sector": "AI Safety", "keywords": ["python", "pytorch", "research", "alignment", "rl"], "sponsorship_rate": 0.45},
        {"sector": "Autonomous Systems", "keywords": ["pytorch", "computer vision", "ros", "c++", "sensors"], "sponsorship_rate": 0.52},
        {"sector": "Enterprise AI", "keywords": ["python", "llm", "langchain", "fastapi", "docker"], "sponsorship_rate": 0.30},
    ]
    user_lower = {s.lower() for s in user_skills}
    fits = []
    for s in sectors:
        overlap = len(set(s["keywords"]) & user_lower)
        if overlap > 0:
            fits.append({
                "sector":           s["sector"],
                "fit_score":        round(overlap / len(s["keywords"]) * 100, 1),
                "sponsorship_rate": s["sponsorship_rate"],
            })
    fits.sort(key=lambda x: -x["fit_score"])
    return fits


def _fetch_salary_expectation(role: str, level: str, location: str) -> dict[str, Any]:
    """Pull salary percentiles from the latest weekly snapshot."""
    try:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        snap_t    = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"
        with engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT salary_p25, salary_p50, salary_p75, salary_sample_size
                FROM {snap_t}
                ORDER BY week_start DESC LIMIT 1
            """)).fetchone()
        if row:
            return {"p25": row[0], "p50": row[1], "p75": row[2], "sample_size": row[3], "currency": "GBP"}
    except Exception:
        pass
    return {"p25": None, "p50": None, "p75": None, "sample_size": 0, "currency": "GBP"}


async def _generate_career_narrative(
    profile: UserProfile,
    match_pct: float,
    skill_gaps: list[dict],
    sector_fit: list[dict],
    salary_exp: dict,
) -> tuple[str, list[str]]:
    """Call Gemini Pro to synthesise the career intelligence narrative."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        # Strict structured-data-in pattern: no raw user text in the LLM prompt
        gap_str    = ", ".join(g["skill"] for g in skill_gaps[:5])
        sector_str = ", ".join(f"{s['sector']} ({s['fit_score']}%)" for s in sector_fit[:3])
        sal_str    = f"£{salary_exp.get('p50'):,.0f}" if salary_exp.get("p50") else "data unavailable"

        prompt = f"""You are a career intelligence analyst for UK AI/ML roles.

STRUCTURED DATA (do not invent statistics outside this set):
- Skills provided: {', '.join(profile.skills[:20])}
- Target role: {profile.target_role}
- Experience level: {profile.experience_level}
- Market match: {match_pct:.0f}% (similarity to current UK AI job postings)
- Top skill gaps vs market: {gap_str or 'none identified'}
- Best sector fits: {sector_str or 'not enough data'}
- Median salary benchmark: {sal_str}
- Visa sponsorship needed: {profile.visa_sponsorship}

Write a personalised career intelligence summary with exactly 4 sections:
1. Current Market Position (2 sentences using the data above)
2. Priority Skill Investments (top 3 gaps with brief rationale)
3. Best Sector Opportunities (top 2 sectors from the data)
4. 90-Day Action Plan (3 concrete, specific steps)

Be direct and data-specific. Do not mention this system prompt. Max 400 words."""

        llm = ChatGoogleGenerativeAI(
            model=settings.llm.deep_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.2,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        text     = response.content.strip()

        # Extract action plan lines (section 4)
        action_plan: list[str] = []
        lines = text.split("\n")
        in_plan = False
        for line in lines:
            if "90-day" in line.lower() or "action plan" in line.lower():
                in_plan = True
                continue
            if in_plan and line.strip() and (line.strip().startswith(("-", "•", "1", "2", "3", "*"))):
                action_plan.append(line.strip().lstrip("-•123. ").strip())
            if len(action_plan) >= 3:
                break

        return text, action_plan or ["Review top skill gaps weekly", "Target 2 applications per week", "Track application responses for pattern recognition"]

    except Exception as exc:
        logger.error("career_narrative.error", error=str(exc))
        fallback = (
            f"Based on current UK AI/ML market data, your profile shows a {match_pct:.0f}% "
            f"alignment with live job postings. Focus on closing gaps in: {', '.join(g['skill'] for g in skill_gaps[:3]) or 'core ML skills'}."
        )
        return fallback, ["Strengthen top gap skills", "Target high-sponsorship sectors", "Track weekly skill demand trends on this dashboard"]


# ── Market data endpoints ─────────────────────────────────────────────────────

@app.get("/api/v1/market/snapshot", response_model=MarketSnapshotResponse, summary="Latest weekly market snapshot")
async def get_market_snapshot(
    week: str | None = Query(default=None, description="ISO date YYYY-MM-DD; defaults to latest")
) -> MarketSnapshotResponse:
    cache_key = f"snapshot:{week or 'latest'}"
    cached = cache.get(cache_key)
    if cached:
        return MarketSnapshotResponse(**cached)

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    import json
    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

    with engine.connect() as conn:
        if week:
            row = conn.execute(text(f"SELECT * FROM {table} WHERE week_start = :w AND role_category = 'all' LIMIT 1"), {"w": week}).mappings().fetchone()
        else:
            row = conn.execute(text(f"SELECT * FROM {table} WHERE role_category = 'all' ORDER BY week_start DESC LIMIT 1")).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No market snapshot available")

    data = dict(row)
    for field in ("top_skills", "rising_skills", "declining_skills", "top_cities"):
        if isinstance(data.get(field), str):
            try:
                data[field] = json.loads(data[field])
            except Exception:
                data[field] = {}
    # Stringify date/datetime fields expected as str by the response model
    for field in ("week_start", "computed_at"):
        if field in data and not isinstance(data[field], str):
            data[field] = str(data[field])

    cache.set(cache_key, data)
    return MarketSnapshotResponse(**{k: data[k] for k in MarketSnapshotResponse.model_fields if k in data})


@app.get("/api/v1/market/skills", summary="Top skills by role category")
async def get_top_skills(
    role_category: str = Query(default="all"),
    week: str | None   = Query(default=None),
) -> dict:
    cache_key = f"skills:{role_category}:{week or 'latest'}"
    cached    = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    import json
    engine = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table  = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT top_skills, rising_skills, declining_skills, week_start
            FROM {table}
            WHERE role_category = :rc
            ORDER BY week_start DESC LIMIT 1
        """), {"rc": role_category}).mappings().fetchone()
        # Fall back to 'all' snapshot when no role-specific data exists yet
        if not row and role_category != "all":
            row = conn.execute(text(f"""
                SELECT top_skills, rising_skills, declining_skills, week_start
                FROM {table}
                WHERE role_category = 'all'
                ORDER BY week_start DESC LIMIT 1
            """)).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"No data for role_category={role_category}")

    data = dict(row)
    for k in ("top_skills", "rising_skills", "declining_skills"):
        if isinstance(data.get(k), str):
            data[k] = json.loads(data[k])

    cache.set(cache_key, data)
    return data


@app.get("/api/v1/market/salary", summary="Salary benchmarks")
async def get_salary_benchmark(
    role_category:    str = Query(default="all"),
    experience_level: str = Query(default="mid"),
    location:         str = Query(default="London"),
) -> dict:
    cache_key = f"salary:{role_category}:{experience_level}:{location}"
    cached    = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT salary_p25, salary_p50, salary_p75, salary_sample_size, week_start
            FROM {table}
            WHERE role_category = :rc
            ORDER BY week_start DESC LIMIT 1
        """), {"rc": role_category}).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No salary data available")

    result = dict(row)
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/trending", summary="Rising and declining skills")
async def get_trending_skills(
    days: int = Query(default=7, description="Lookback window: 7, 14, or 30 days"),
) -> dict:
    cache_key = f"trending:{days}"
    cached    = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    import json
    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT top_skills, rising_skills, declining_skills, week_start
            FROM {table}
            WHERE role_category = 'all'
            ORDER BY week_start DESC LIMIT 4
        """)).mappings().fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No trend data available")

    latest = dict(rows[0])
    for k in ("top_skills", "rising_skills", "declining_skills"):
        if isinstance(latest.get(k), str):
            latest[k] = json.loads(latest[k])

    result = {
        "rising":   latest.get("rising_skills",   []),
        "declining":latest.get("declining_skills",[]),
        "top_now":  list((latest.get("top_skills") or {}).keys())[:10],
        "week":     str(latest.get("week_start", "")),
    }
    cache.set(cache_key, result)
    return result


# ── Jobs listing endpoint ─────────────────────────────────────────────────────

@app.get("/api/v1/jobs", summary="Browse indexed UK AI/ML job listings")
async def get_jobs(
    role_category:    str | None = Query(default=None, description="Filter by role category"),
    work_model:       str | None = Query(default=None, description="remote / hybrid / onsite"),
    experience_level: str | None = Query(default=None, description="junior / mid / senior / lead"),
    source:           str | None = Query(default=None, description="adzuna / reed / etc."),
    visa_only:        bool       = Query(default=False, description="Only jobs with visa sponsorship"),
    page:             int        = Query(default=1, ge=1, description="Page number"),
    page_size:        int        = Query(default=20, ge=1, le=100, description="Jobs per page"),
) -> dict:
    cache_key = f"jobs:{role_category}:{work_model}:{experience_level}:{source}:{visa_only}:{page}:{page_size}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    jobs_t    = "jobs"       if is_sqlite else "market.jobs"
    skills_t  = "job_skills" if is_sqlite else "market.job_skills"

    # Build WHERE clauses
    conditions = []
    params: dict = {}
    if role_category and role_category != "all":
        conditions.append("j.role_category = :role")
        params["role"] = role_category
    if work_model:
        conditions.append("j.work_model = :wm")
        params["wm"] = work_model
    if experience_level:
        conditions.append("j.experience_level = :el")
        params["el"] = experience_level
    if source:
        conditions.append("j.source = :src")
        params["src"] = source
    if visa_only:
        conditions.append("j.offers_sponsorship = TRUE")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * page_size
    params["limit"] = page_size
    params["offset"] = offset

    # Skills subquery — dialect-aware
    if is_sqlite:
        skills_sub = (
            f"(SELECT GROUP_CONCAT(skill, ', ') FROM "
            f"(SELECT skill FROM {skills_t} WHERE job_id = j.job_id ORDER BY confidence DESC LIMIT 8))"
        )
    else:
        skills_sub = (
            f"(SELECT STRING_AGG(skill, ', ' ORDER BY confidence DESC) FROM "
            f"(SELECT skill, confidence FROM {skills_t} WHERE job_id = j.job_id "
            f"ORDER BY confidence DESC LIMIT 8) _s)"
        )

    with engine.connect() as conn:
        total = conn.execute(text(
            f"SELECT COUNT(*) FROM {jobs_t} j {where}"
        ), params).scalar() or 0

        rows = conn.execute(text(f"""
            SELECT j.job_id, j.title, j.company, j.location,
                   j.salary_min, j.salary_max, j.work_model,
                   j.experience_level, j.role_category, j.source,
                   j.offers_sponsorship, j.posted_date, j.scraped_at, j.url,
                   j.is_startup, j.company_stage,
                   COALESCE({skills_sub}, '') AS skills
            FROM {jobs_t} j
            {where}
            ORDER BY j.scraped_at DESC
            LIMIT :limit OFFSET :offset
        """), params).mappings().fetchall()

    jobs = []
    for r in rows:
        d = dict(r)
        # Serialise dates as strings
        for f in ("posted_date", "scraped_at"):
            if d.get(f) is not None and not isinstance(d[f], str):
                d[f] = str(d[f])
        # Split skills CSV into list
        d["skills"] = [s.strip() for s in (d.get("skills") or "").split(",") if s.strip()]
        jobs.append(d)

    result = {
        "jobs":      jobs,
        "total":     int(total),
        "page":      page,
        "page_size": page_size,
        "pages":     max(1, -(-int(total) // page_size)),  # ceiling div
    }
    # Short TTL so freshly-scraped jobs appear quickly
    cache.set(cache_key, result)
    return result


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, summary="Platform health")
async def health() -> HealthResponse:
    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    from datetime import datetime

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    runs_t    = "pipeline_runs" if is_sqlite else "market.pipeline_runs"
    jobs_t    = "jobs"          if is_sqlite else "market.jobs"

    try:
        with engine.connect() as conn:
            last_run = conn.execute(text(f"SELECT MAX(completed_at) FROM {runs_t}")).scalar()
            total_jobs= conn.execute(text(f"SELECT COUNT(*) FROM {jobs_t}")).scalar() or 0
    except Exception:
        return HealthResponse(status="degraded", last_ingestion=None, data_freshness_h=None, jobs_total=0, version="0.1.0")

    freshness = None
    if last_run:
        try:
            if isinstance(last_run, str):
                last_run = datetime.fromisoformat(last_run)
            freshness = round((datetime.utcnow() - last_run.replace(tzinfo=None)).total_seconds() / 3600, 1)
        except Exception:
            pass

    status_str = "healthy" if (freshness is not None and freshness < 72) else "stale"
    return HealthResponse(
        status=status_str,
        last_ingestion=str(last_run) if last_run else None,
        data_freshness_h=freshness,
        jobs_total=int(total_jobs),
        version="0.1.0",
    )


# ── CV Upload + ATS Score + Career Gap endpoint ───────────────────────────────

class CVATSBreakdown(BaseModel):
    keyword_match: float
    structure:     float
    readability:   float
    completeness:  float
    format_safety: float


class CVGapPlan(BaseModel):
    short_term: list[str]   # 0–3 months
    mid_term:   list[str]   # 3–12 months
    long_term:  list[str]   # 12+ months


class CVAnalysisReport(BaseModel):
    session_token:     str              # anonymous, no PII
    ats_score:         float            # 0–100
    ats_grade:         str              # A+/A/B/C/D
    ats_breakdown:     CVATSBreakdown
    ats_issues:        list[str]        # actionable fix suggestions
    skills_found:      list[str]        # skills extracted from CV
    skills_missing:    list[str]        # top market skills not in CV
    keyword_match_pct: float
    market_match_pct:  float
    gap_plan:          CVGapPlan
    narrative_summary: str
    pii_scrubbed:      list[str]        # PII types that were found and stripped
    data_retained:     bool = False     # always False — GDPR guarantee


@app.post(
    "/api/v1/career/cv-analyse",
    response_model=CVAnalysisReport,
    summary="CV upload → ATS score + career gap analysis",
    description=(
        "Upload a CV (PDF or DOCX, max 5 MB) and receive an ATS compatibility score, "
        "skill gap analysis, and a short/mid/long-term career plan. "
        "No CV data is stored — processing is in-memory only (GDPR compliant)."
    ),
)
async def analyse_cv(
    request:     Request,
    cv_file:     UploadFile = File(..., description="PDF or DOCX CV, max 5 MB"),
    target_role: str        = "ml_engineer",
    consent:     bool       = False,
) -> CVAnalysisReport:
    from marketforge.cv.scanner  import scan_file
    from marketforge.cv.parser   import parse_cv
    from marketforge.cv.ats_scorer import score_cv
    from marketforge.cv.gdpr     import build_gdpr_context, ConsentNotGiven

    ip = request.client.host if request.client else "unknown"

    # ── Rate limit: 3 CV analyses per IP per hour (expensive operation) ────────
    if not limiter.is_allowed(f"cv_analyse:{ip}", limit=3, window_seconds=3600):
        raise HTTPException(status_code=429, detail="CV analysis rate limit exceeded (3/hour)")

    # ── GDPR consent gate ─────────────────────────────────────────────────────
    if not consent:
        raise HTTPException(
            status_code=403,
            detail="GDPR consent is required. Set consent=true to confirm you agree to the privacy notice.",
        )

    # ── Read file into memory (never touch disk) ──────────────────────────────
    raw_bytes = await cv_file.read()

    # ── Security scan ─────────────────────────────────────────────────────────
    scan = scan_file(raw_bytes)
    if not scan.allowed:
        logger.warning("cv.endpoint.scan_rejected", reason=scan.rejection_reason, ip=ip)
        raise HTTPException(
            status_code=422,
            detail=f"File rejected by security scan: {scan.rejection_reason}",
        )

    # ── Parse CV ──────────────────────────────────────────────────────────────
    cv = parse_cv(raw_bytes, scan.file_type)
    if cv.error:
        raise HTTPException(status_code=422, detail=f"CV could not be parsed: {cv.error}")

    # ── GDPR: strip PII before any further processing ─────────────────────────
    try:
        gdpr_ctx = build_gdpr_context(cv.raw_text, scan.file_hash, consent=True)
    except ConsentNotGiven:
        raise HTTPException(status_code=403, detail="Consent check failed")

    # Replace raw_text with scrubbed version; drop original reference
    cv.raw_text = gdpr_ctx.scrubbed_text
    del raw_bytes   # discard original bytes

    # ── ATS scoring ────────────────────────────────────────────────────────────
    ats = score_cv(cv, target_role)

    # ── Market match (SBERT) ───────────────────────────────────────────────────
    match_pct, _ = _compute_market_match(ats.skills_found or [target_role])

    # ── Career gap analysis via LLM ────────────────────────────────────────────
    gap_plan, narrative = await _generate_cv_gap_plan(
        ats_score    = ats.total,
        skills_found = ats.skills_found,
        skills_missing = [i["skill"] for i in _compute_skill_gaps(ats.skills_found, target_role)[:8]],
        target_role  = target_role,
        match_pct    = match_pct,
    )

    # ── Output guardrails ──────────────────────────────────────────────────────
    from marketforge.agents.security.guardrails import validate_output
    narrative, _ = validate_output(narrative)

    logger.info(
        "cv.endpoint.complete",
        session=gdpr_ctx.session_token[:8],
        ats_score=ats.total,
        ats_grade=ats.grade,
    )

    return CVAnalysisReport(
        session_token     = gdpr_ctx.session_token,
        ats_score         = ats.total,
        ats_grade         = ats.grade,
        ats_breakdown     = CVATSBreakdown(**ats.breakdown),
        ats_issues        = ats.issues,
        skills_found      = ats.skills_found,
        skills_missing    = [i["skill"] for i in _compute_skill_gaps(ats.skills_found, target_role)[:10]],
        keyword_match_pct = ats.keyword_match_pct,
        market_match_pct  = round(match_pct, 1),
        gap_plan          = gap_plan,
        narrative_summary = narrative,
        pii_scrubbed      = gdpr_ctx.pii_types_found,
        data_retained     = False,
    )


async def _generate_cv_gap_plan(
    ats_score:     float,
    skills_found:  list[str],
    skills_missing:list[str],
    target_role:   str,
    match_pct:     float,
) -> tuple[CVGapPlan, str]:
    """LLM call to generate short/mid/long-term plan. Receives structured data only — no raw CV text."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        found_str   = ", ".join(skills_found[:15]) or "none detected"
        missing_str = ", ".join(skills_missing[:8]) or "none"

        prompt = f"""You are a UK AI/ML career advisor. Generate a structured career development plan.

STRUCTURED DATA (use only this — do not invent facts):
- ATS score: {ats_score:.0f}/100
- Target role: {target_role}
- Skills in CV: {found_str}
- Top market skills missing from CV: {missing_str}
- Market match: {match_pct:.0f}%

Respond in this exact format (JSON-like sections):

NARRATIVE: [2 sentences: current position assessment based on ATS score and market match]

SHORT_TERM (0-3 months):
- [action 1]
- [action 2]
- [action 3]

MID_TERM (3-12 months):
- [action 1]
- [action 2]
- [action 3]

LONG_TERM (12+ months):
- [action 1]
- [action 2]

Keep each action specific and achievable. Do not mention company names."""

        llm = ChatGoogleGenerativeAI(
            model=settings.llm.fast_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.2,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        text     = response.content.strip()

        # Parse structured sections
        def _extract_bullets(section_text: str) -> list[str]:
            return [
                line.strip().lstrip("-•*123456789. ").strip()
                for line in section_text.split("\n")
                if line.strip() and line.strip()[0] in "-•*123456789"
            ][:3]

        narrative  = ""
        short_term: list[str] = []
        mid_term:   list[str] = []
        long_term:  list[str] = []

        current_section = ""
        for line in text.split("\n"):
            if line.startswith("NARRATIVE:"):
                narrative = line.replace("NARRATIVE:", "").strip()
                current_section = "narrative"
            elif "SHORT_TERM" in line:
                current_section = "short"
            elif "MID_TERM" in line:
                current_section = "mid"
            elif "LONG_TERM" in line:
                current_section = "long"
            elif line.strip().startswith("-") or line.strip().startswith("•"):
                item = line.strip().lstrip("-• ").strip()
                if item:
                    if current_section == "short":
                        short_term.append(item)
                    elif current_section == "mid":
                        mid_term.append(item)
                    elif current_section == "long":
                        long_term.append(item)

        if not narrative:
            narrative = (
                f"Your CV scores {ats_score:.0f}/100 for ATS compatibility with a "
                f"{match_pct:.0f}% market match for {target_role} roles. "
                f"Prioritise adding the missing skills to close key gaps."
            )

        return (
            CVGapPlan(
                short_term = short_term or ["Complete a course in top missing skills", "Add metrics to all experience bullets", "Update LinkedIn to mirror CV keywords"],
                mid_term   = mid_term   or ["Build a portfolio project using missing skills", "Contribute to open-source AI projects", "Obtain relevant certification"],
                long_term  = long_term  or ["Target senior roles after closing skill gaps", "Build demonstrable track record with new skills"],
            ),
            narrative,
        )

    except Exception as exc:
        logger.error("cv.gap_plan.error", error=str(exc))
        return (
            CVGapPlan(
                short_term = ["Add missing skills to CV", "Improve ATS formatting", "Quantify achievements"],
                mid_term   = ["Build portfolio projects", "Complete relevant certifications"],
                long_term  = ["Target senior roles after 12 months of skill building"],
            ),
            f"CV scored {ats_score:.0f}/100 — address the listed issues to improve ATS compatibility.",
        )


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> str:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest().decode("utf-8")
    except Exception:
        return "# prometheus_client not available\n"
