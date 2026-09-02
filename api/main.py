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

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from marketforge.agents.security.guardrails import validate_input
from marketforge.config.settings import settings
from marketforge.memory.postgres import init_database
from marketforge.memory.redis_cache import DashboardCache, RateLimiter
from marketforge.utils.logger import setup_logging
from api.security import SecurityMiddleware

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
    # Pre-warm SBERT so both gunicorn workers have the model loaded before
    # handling requests — prevents 30s+ cold-start on the first CV analysis.
    try:
        await asyncio.to_thread(_get_sbert)
        logger.info("sbert.warmed")
    except Exception as exc:
        logger.warning("sbert.warm_failed", error=str(exc))
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

_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()] or ["*"]

# CORSMiddleware must be registered BEFORE SecurityMiddleware so it becomes the
# inner layer in the ASGI stack. FastAPI applies middleware LIFO, so the last
# add_middleware call becomes outermost. SecurityMiddleware uses BaseHTTPMiddleware
# which reconstructs the ASGI response and breaks CORSMiddleware's header injection
# when SecurityMiddleware is the inner layer.
#
# Correct ASGI stack (request path):
#   SecurityMiddleware (outer) → CORSMiddleware (inner) → route handlers
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)
app.add_middleware(SecurityMiddleware)


# ── IP extraction utility ─────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """Railway appends the real client IP as the rightmost X-Forwarded-For entry."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        return parts[-1] if parts else "unknown"
    return request.client.host if request.client else "unknown"


# ── Rate limit middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip  = _get_client_ip(request)
    path = request.url.path

    # CV analyse has its own per-endpoint limiter (3/hour) — skip middleware check
    if path == "/api/v1/career/cv-analyse":
        pass
    # Career advisor: 10 req/min (LLM-backed, expensive)
    elif path.startswith("/api/v1/career"):
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
async def analyse_career(profile: UserProfile, request: Request, fastapi_response: Response) -> CareerIntelligenceReport:
    fastapi_response.headers["Access-Control-Allow-Origin"] = "*"
    ip = _get_client_ip(request)

    # ── Security gate ─────────────────────────────────────────────────────────
    all_text = " ".join(profile.skills) + " " + profile.target_role + " " + (profile.free_text or "")
    sec_result = validate_input(all_text, field_name="profile", source_ip=ip)
    if not sec_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=sec_result.rejection_reason,
        )

    skills_text = sec_result.sanitised_text

    # ── Market match via SBERT + ChromaDB ────────────────────────────────────
    match_pct, match_dist = await asyncio.to_thread(_compute_market_match, profile.skills, profile.target_role)

    # ── Skill gap analysis ────────────────────────────────────────────────────
    skill_gaps            = await asyncio.to_thread(_compute_skill_gaps, profile.skills, profile.target_role)

    # ── Sector fit ────────────────────────────────────────────────────────────
    sector_fit = _compute_sector_fit(profile.skills)

    # ── Salary expectation ────────────────────────────────────────────────────
    salary_exp = await asyncio.to_thread(_fetch_salary_expectation, profile.target_role, profile.experience_level, profile.location)

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


def _compute_market_match(
    skills:      list[str],
    target_role: str = "",
) -> tuple[float, dict[str, float]]:
    """
    SBERT embed the skill list and compare against job descriptions for the
    target role.  When target_role is provided, only jobs with a matching
    role_category are sampled so the score reflects fit for that role.
    """
    try:
        import numpy as np
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        from marketforge.cv.ats_scorer import _normalise_role

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        table     = "jobs" if is_sqlite else "market.jobs"
        role_cat  = _normalise_role(target_role) if target_role else ""

        with engine.connect() as conn:
            if role_cat and role_cat != "other":
                # Role-specific sample — try up to 300 to get enough rows
                rows = conn.execute(text(f"""
                    SELECT title, role_category FROM {table}
                    WHERE role_category = :role
                    ORDER BY scraped_at DESC LIMIT 300
                """), {"role": role_cat}).fetchall()
                # Fall back to all roles if insufficient data for this role
                if len(rows) < 20:
                    rows = conn.execute(text(f"""
                        SELECT title, role_category FROM {table}
                        ORDER BY scraped_at DESC LIMIT 200
                    """)).fetchall()
            else:
                rows = conn.execute(text(f"""
                    SELECT title, role_category FROM {table}
                    ORDER BY scraped_at DESC LIMIT 200
                """)).fetchall()

        if not rows:
            return 50.0, {"strong": 0.3, "moderate": 0.4, "weak": 0.3}

        model       = _get_sbert()
        profile_emb = model.encode(" ".join(skills), normalize_embeddings=True)
        job_texts   = [f"{r[0]} {r[1] or ''}" for r in rows]
        job_embs    = model.encode(job_texts, normalize_embeddings=True, batch_size=64)

        similarities = np.dot(job_embs, profile_emb)
        strong    = float(np.mean(similarities > 0.75))
        moderate  = float(np.mean((similarities > 0.55) & (similarities <= 0.75)))
        weak      = float(np.mean(similarities <= 0.55))
        match_pct = float(np.mean(similarities) * 100)

        return min(max(match_pct, 0), 100), {
            "strong":   round(strong,   3),
            "moderate": round(moderate, 3),
            "weak":     round(weak,     3),
        }
    except Exception as exc:
        logger.warning("market_match.error", error=str(exc))
        return 50.0, {"strong": 0.3, "moderate": 0.4, "weak": 0.3}


def _compute_skill_gaps(user_skills: list[str], target_role: str) -> list[dict[str, Any]]:
    """
    Compare user skills against top-demanded skills for the target role.
    Queries live job_skills filtered by role_category; falls back to the
    global weekly snapshot if no role-specific data exists.
    """
    try:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        import json
        from marketforge.cv.ats_scorer import _normalise_role

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        jobs_t    = "jobs"       if is_sqlite else "market.jobs"
        skills_t  = "job_skills" if is_sqlite else "market.job_skills"
        snap_t    = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"
        role_cat  = _normalise_role(target_role)
        user_lower = {s.lower() for s in user_skills}

        top_skills: dict[str, int] = {}

        with engine.connect() as conn:
            # Live per-role query
            rows = conn.execute(text(f"""
                SELECT js.skill, COUNT(*) AS cnt
                FROM {skills_t} js
                JOIN {jobs_t} j ON j.job_id = js.job_id
                WHERE j.role_category = :role
                GROUP BY js.skill
                ORDER BY cnt DESC
                LIMIT 50
            """), {"role": role_cat}).fetchall()

            if rows:
                top_skills = {r[0]: r[1] for r in rows}
            else:
                # Snapshot fallback
                row = conn.execute(text(f"""
                    SELECT top_skills FROM {snap_t}
                    WHERE role_category = 'all'
                    ORDER BY week_start DESC LIMIT 1
                """)).fetchone()
                if row and row[0]:
                    top_skills = json.loads(row[0]) if isinstance(row[0], str) else row[0]

        gaps = []
        for skill, count in sorted(top_skills.items(), key=lambda x: -x[1]):
            if skill.lower() not in user_lower:
                gaps.append({
                    "skill":         skill,
                    "market_demand": count,
                    "priority":      "high" if count > 50 else "medium",
                })
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


@app.get("/api/v1/market/snapshot-history", summary="Weekly snapshot history (real time series)")
async def get_snapshot_history(
    weeks: int = Query(default=26, ge=1, le=104, description="How many recent weeks to return"),
) -> dict:
    cache_key = f"snapshot_history:{weeks}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT week_start, job_count, salary_p50, sponsorship_rate
            FROM {table}
            WHERE role_category = 'all'
            ORDER BY week_start DESC LIMIT :n
        """), {"n": weeks}).fetchall()

    result = {
        "role_category": "all",
        "weeks": [
            {"week_start": str(ws), "job_count": jc, "salary_p50": p50, "sponsorship_rate": sr}
            for ws, jc, p50, sr in reversed(rows)
        ],
    }
    cache.set(cache_key, result)
    return result


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
    experience_level: str = Query(default="all"),
    location:         str = Query(default="all"),
) -> dict:
    cache_key = f"salary:{role_category}:{experience_level}:{location}"
    cached    = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"

    # Fast path: use precomputed snapshot when no experience/location filter
    if experience_level in ("all", "") and location in ("all", ""):
        table = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"
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
    else:
        result = _compute_salary_from_jobs(engine, is_sqlite, role_category, experience_level, location)
        if not result:
            raise HTTPException(status_code=404, detail="No salary data available")

    cache.set(cache_key, result)
    return result


def _compute_salary_from_jobs(
    engine,
    is_sqlite: bool,
    role_category: str,
    experience_level: str,
    location: str,
) -> dict | None:
    from sqlalchemy import text
    from datetime import date

    table = "jobs" if is_sqlite else "market.jobs"
    conditions = ["salary_min IS NOT NULL"]
    params: dict = {}

    if role_category and role_category != "all":
        conditions.append("role_category = :rc")
        params["rc"] = role_category

    if experience_level and experience_level not in ("all", ""):
        _EXP_MAP: dict[str, tuple] = {
            "junior":    ("junior",),
            "mid":       ("mid", "mid-level"),
            "senior":    ("senior",),
            "principal": ("lead", "principal", "staff"),
            "lead":      ("lead", "principal", "staff"),
        }
        levels = _EXP_MAP.get(experience_level.lower(), (experience_level.lower(),))
        placeholders = ", ".join(f":el{i}" for i in range(len(levels)))
        conditions.append(f"LOWER(experience_level) IN ({placeholders})")
        for i, lv in enumerate(levels):
            params[f"el{i}"] = lv

    if location and location not in ("all", ""):
        if is_sqlite:
            conditions.append("location LIKE :loc")
        else:
            conditions.append("location ILIKE :loc")
        params["loc"] = f"%{location}%"

    where = " AND ".join(conditions)

    try:
        with engine.connect() as conn:
            if is_sqlite:
                rows = conn.execute(text(f"""
                    SELECT (salary_min + COALESCE(salary_max, salary_min)) / 2.0 AS mid_sal
                    FROM {table}
                    WHERE {where}
                    ORDER BY mid_sal
                """), params).fetchall()
                salaries = [r[0] for r in rows if r[0] is not None]
                if not salaries:
                    return None
                n = len(salaries)
                return {
                    "salary_p25": salaries[max(0, int(n * 0.25))],
                    "salary_p50": salaries[max(0, int(n * 0.50))],
                    "salary_p75": salaries[min(n - 1, int(n * 0.75))],
                    "salary_sample_size": n,
                    "week_start": str(date.today()),
                }
            else:
                row = conn.execute(text(f"""
                    SELECT
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY mid_sal) AS salary_p25,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY mid_sal) AS salary_p50,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY mid_sal) AS salary_p75,
                        COUNT(*) AS salary_sample_size
                    FROM (
                        SELECT (salary_min + COALESCE(salary_max, salary_min)) / 2.0 AS mid_sal
                        FROM {table}
                        WHERE {where}
                    ) t
                    WHERE mid_sal IS NOT NULL
                """), params).mappings().fetchone()
                if not row or row["salary_p50"] is None:
                    return None
                result = dict(row)
                result["week_start"] = str(date.today())
                return result
    except Exception as exc:
        logger.warning("salary_from_jobs.error", error=str(exc))
        return None


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


# ── Market detail endpoints ───────────────────────────────────────────────────

def _extract_city(location: str) -> str:
    """Normalise 'London, UK' → 'London', 'Greater Manchester' → 'Manchester'."""
    if not location:
        return ""
    city = location.split(",")[0].strip()
    _ALIASES = {
        "Greater London": "London", "City of London": "London",
        "East London": "London", "North London": "London",
        "South London": "London", "West London": "London",
        "Central London": "London", "London Area": "London",
        "Greater Manchester": "Manchester", "Edinburgh City": "Edinburgh",
        "City of Edinburgh": "Edinburgh", "City of Bristol": "Bristol",
        "City of Birmingham": "Birmingham",
    }
    return _ALIASES.get(city, city)


@app.get("/api/v1/market/hiring-velocity", summary="Hiring velocity by role")
async def get_hiring_velocity() -> dict:
    cache_key = "hiring_velocity"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    from collections import defaultdict

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

    _ROLE_DISPLAY = {
        "ml_engineer":    "ML Engineer",
        "ai_engineer":    "AI / LLM Engineer",
        "mlops_engineer": "MLOps / Platform",
        "ai_safety":      "AI Safety Engineer",
        "data_scientist": "Data Scientist",
        "data_analyst":   "Data Analyst",
        "nlp_engineer":   "NLP Engineer",
        "data_engineer":  "Data Engineer",
        "cv_engineer":    "Computer Vision Eng.",
        "ai_researcher":  "AI Researcher",
    }

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT role_category, job_count, week_start
            FROM {table}
            WHERE role_category != 'all'
            ORDER BY week_start DESC
            LIMIT 30
        """)).mappings().fetchall()

    role_weeks: dict = defaultdict(list)
    for r in rows:
        role_weeks[r["role_category"]].append(
            {"job_count": r["job_count"] or 0, "week_start": str(r["week_start"])}
        )

    velocity = []
    for role, weeks in role_weeks.items():
        if len(weeks) >= 2:
            current  = weeks[0]["job_count"]
            previous = weeks[1]["job_count"]
            growth_pct = round((current - previous) / max(previous, 1) * 100, 1)
            direction  = "up" if growth_pct >= 0 else "down"
        else:
            growth_pct = 0.0
            direction  = "neutral"
        velocity.append({
            "role":          _ROLE_DISPLAY.get(role, role.replace("_", " ").title()),
            "role_category": role,
            "growth_pct":    growth_pct,
            "direction":     direction,
        })

    velocity.sort(key=lambda x: -abs(x["growth_pct"]))
    result = {"velocity": velocity}
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/cities", summary="Top UK hiring cities")
async def get_cities() -> dict:
    cache_key = "cities"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    from datetime import date, timedelta

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "jobs" if is_sqlite else "market.jobs"

    today      = date.today()
    week_start = today - timedelta(days=today.weekday())

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT location, COUNT(*) AS cnt
            FROM {table}
            WHERE location IS NOT NULL
              AND scraped_at >= :ws
            GROUP BY location
            ORDER BY cnt DESC
            LIMIT 200
        """), {"ws": str(week_start)}).fetchall()

    city_counts: dict[str, int] = {}
    for location, cnt in rows:
        city = _extract_city(location)
        if city:
            city_counts[city] = city_counts.get(city, 0) + cnt

    sorted_cities = sorted(city_counts.items(), key=lambda x: -x[1])[:10]
    cities = [{"city": c, "job_count": n} for c, n in sorted_cities]

    result = {"cities": cities, "week_start": str(week_start)}
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/company-mix", summary="Company type mix")
async def get_company_mix() -> dict:
    cache_key = "company_mix"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "jobs" if is_sqlite else "market.jobs"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT company_stage, is_startup, COUNT(*) AS cnt
            FROM {table}
            GROUP BY company_stage, is_startup
        """)).fetchall()
        total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 1

    buckets: dict[str, int] = {
        "Scale-up (50–500)":   0,
        "Enterprise (500+)":   0,
        "Startup (<50)":       0,
        "Research / Academic": 0,
    }
    for stage, is_startup, cnt in rows:
        sl = (stage or "").lower()
        if any(k in sl for k in ("enterprise", "large", "corporate", "public")):
            buckets["Enterprise (500+)"] += cnt
        elif any(k in sl for k in ("research", "academic", "university", "institute")):
            buckets["Research / Academic"] += cnt
        elif any(k in sl for k in ("scale", "growth", "mid", "series b", "series c")):
            buckets["Scale-up (50–500)"] += cnt
        elif any(k in sl for k in ("startup", "early", "seed", "series a")) or is_startup:
            buckets["Startup (<50)"] += cnt
        else:
            buckets["Scale-up (50–500)"] += cnt  # default unknown to scale-up

    mix = [
        {"type": t, "pct": round(n / max(total, 1) * 100, 1), "job_count": n}
        for t, n in buckets.items()
    ]
    result = {"mix": mix}
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/sponsorship-by-sector", summary="Visa sponsorship rates by sector")
async def get_sponsorship_by_sector() -> dict:
    cache_key = "sponsorship_by_sector"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "jobs" if is_sqlite else "market.jobs"

    _SECTOR_MAP = {
        "ai_safety":      "AI Safety",
        "cv_engineer":    "Autonomous Systems",
        "ai_engineer":    "Autonomous Systems",
        "nlp_engineer":   "FinTech AI",
        "data_analyst":   "FinTech AI",
        "data_scientist": "HealthTech AI",
        "ml_engineer":    "Enterprise AI",
        "mlops_engineer": "Enterprise AI",
        "data_engineer":  "Enterprise AI",
        "ai_researcher":  "AI Safety",
    }

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT
                role_category,
                COUNT(*) AS total,
                SUM(CASE WHEN offers_sponsorship THEN 1 ELSE 0 END) AS sponsored
            FROM {table}
            WHERE role_category IS NOT NULL
            GROUP BY role_category
            HAVING COUNT(*) >= 5
        """)).fetchall()

    sector_data: dict[str, dict] = {}
    for role, total, sponsored in rows:
        sector = _SECTOR_MAP.get(role, role.replace("_", " ").title())
        if sector not in sector_data:
            sector_data[sector] = {"total": 0, "sponsored": 0}
        sector_data[sector]["total"]    += int(total or 0)
        sector_data[sector]["sponsored"] += int(sponsored or 0)

    sectors = [
        {
            "sector":           s,
            "sponsorship_rate": round(d["sponsored"] / max(d["total"], 1), 3),
        }
        for s, d in sector_data.items()
    ]
    sectors.sort(key=lambda x: -x["sponsorship_rate"])
    result = {"sectors": sectors}
    cache.set(cache_key, result)
    return result


# ── External trusted-source endpoints (ONS, GOV.UK) ─────────────────────────────
# Populated by worker.py's monthly `external_stats` job — see
# marketforge.agents.research.{ons_vacancy_agent,sponsor_register_agent,ashe_salary_agent}.
# Every response is honestly captioned with its source/methodology since these
# replace figures that used to be presented as real analysis but weren't.

@app.get("/api/v1/market/external/vacancy-trend", summary="ONS national vacancy trend (tech sector proxy)")
async def get_external_vacancy_trend() -> dict:
    cache_key = "external_vacancy_trend"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "external_ons_vacancies" if is_sqlite else "market.external_ons_vacancies"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT month, vacancies_index FROM {table}
            ORDER BY month ASC
        """)).fetchall()

    result = {
        "source": "ONS, dataset LMS, CDID JP9P",
        "series_label": "UK Job Vacancies (thousands) — Information & Communication",
        "methodology": (
            "ONS does not publish an AI/ML-specific vacancy series. This is the "
            "closest official proxy (the SIC section covering software/tech "
            "employers) and is shown as sector context, not a literal AI-jobs count."
        ),
        "trend": [{"month": m, "vacancies_index": v} for m, v in rows],
    }
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/external/sponsor-verification", summary="Sponsor register verification rate")
async def get_external_sponsor_verification() -> dict:
    cache_key = "external_sponsor_verification"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "external_sponsor_matches" if is_sqlite else "market.external_sponsor_matches"

    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT COUNT(*), SUM(CASE WHEN is_licensed_sponsor THEN 1 ELSE 0 END)
            FROM {table}
        """)).fetchone()

    sample_size = int(row[0] or 0)
    verified    = int(row[1] or 0)
    result = {
        "source": "GOV.UK Register of Licensed Sponsors: Workers",
        "methodology": (
            "Employer names scraped this quarter are normalised and matched "
            "against the official register. verified_pct is the share of "
            "distinct employers that are actually licensed to sponsor visas — "
            "an authoritative alternative to text-based sponsorship claims in "
            "job postings."
        ),
        "sample_size":  sample_size,
        "verified_pct": round(verified / max(sample_size, 1), 3) if sample_size else None,
    }
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/external/salary-benchmark", summary="ONS ASHE salary benchmark by role")
async def get_external_salary_benchmark() -> dict:
    cache_key = "external_salary_benchmark"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "external_ashe_salary" if is_sqlite else "market.external_ashe_salary"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT role_category, soc_code, soc_title, year, salary_p25, salary_p50, salary_p75
            FROM {table}
            WHERE year = (SELECT MAX(year) FROM {table})
        """)).fetchall()

    result = {
        "source": "ONS Annual Survey of Hours and Earnings (ASHE), Table 14",
        "methodology": (
            "ONS does not publish a distinct occupation code for 'Data Scientist', "
            "'ML Engineer', etc. Every role_category below is benchmarked against "
            "the closest official proxy — SOC 2134 'Programmers and software "
            "development professionals' — not a per-role ONS figure."
        ),
        "benchmarks": [
            {
                "role_category": rc, "soc_code": soc, "soc_title": title, "year": yr,
                "salary_p25": p25, "salary_p50": p50, "salary_p75": p75,
            }
            for rc, soc, title, yr, p25, p50, p75 in rows
        ],
    }
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/external/graduate-outcomes", summary="UK graduate employment outcomes + pipeline size")
async def get_graduate_outcomes() -> dict:
    cache_key = "graduate_outcomes"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    emp_table = "external_grad_employment" if is_sqlite else "market.external_grad_employment"
    hc_table  = "external_grad_headcount"  if is_sqlite else "market.external_grad_headcount"

    with engine.connect() as conn:
        emp_row = conn.execute(text(f"""
            SELECT year, employment_rate, hs_employment_rate, unemployment_rate, inactivity_rate
            FROM {emp_table} ORDER BY year DESC LIMIT 1
        """)).fetchone()
        hc_row = conn.execute(text(f"""
            SELECT year, qualifiers_count FROM {hc_table}
            WHERE subject = 'Computing' ORDER BY year DESC LIMIT 1
        """)).fetchone()

    result = {
        "source": "DfE Explore Education Statistics (data.explore-education-statistics.service.gov.uk)",
        "methodology": (
            "employment_rate/unemployment_rate/inactivity_rate are the unweighted "
            "average of the Male and Female breakdown rows — the source dataset "
            "publishes no unsegmented 'all graduates' row. England only, not UK-wide. "
            "qualifiers_count is the number of Computing-subject HE qualifiers "
            "(all levels combined) for that academic year."
        ),
        "employment": {
            "year": emp_row[0], "employment_rate": emp_row[1], "hs_employment_rate": emp_row[2],
            "unemployment_rate": emp_row[3], "inactivity_rate": emp_row[4],
        } if emp_row else None,
        "computing_qualifiers": {
            "academic_year": hc_row[0], "qualifiers_count": hc_row[1],
        } if hc_row else None,
    }
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/entry-level/skill-shift", summary="Skill demand: overall rank vs entry-level rank")
async def get_entry_level_skill_shift() -> dict:
    cache_key = "entry_level_skill_shift"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    from datetime import date, timedelta

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    jobs_t    = "jobs"       if is_sqlite else "market.jobs"
    skills_t  = "job_skills" if is_sqlite else "market.job_skills"

    since = str(date.today() - timedelta(days=90))

    with engine.connect() as conn:
        overall_rows = conn.execute(text(f"""
            SELECT js.skill, COUNT(*) AS cnt
            FROM {skills_t} js JOIN {jobs_t} j ON j.job_id = js.job_id
            WHERE j.scraped_at >= :since
            GROUP BY js.skill ORDER BY cnt DESC LIMIT 40
        """), {"since": since}).fetchall()
        junior_rows = conn.execute(text(f"""
            SELECT js.skill, COUNT(*) AS cnt
            FROM {skills_t} js JOIN {jobs_t} j ON j.job_id = js.job_id
            WHERE j.scraped_at >= :since AND j.experience_level = 'junior'
            GROUP BY js.skill ORDER BY cnt DESC LIMIT 40
        """), {"since": since}).fetchall()

    overall_rank = {skill: i + 1 for i, (skill, _) in enumerate(overall_rows)}
    junior_rank  = {skill: i + 1 for i, (skill, _) in enumerate(junior_rows)}

    shifts = []
    for skill, jr in junior_rank.items():
        orank = overall_rank.get(skill, len(overall_rank) + 5)
        shifts.append({"skill": skill, "overall_rank": orank, "junior_rank": jr, "rank_delta": orank - jr})

    shifts.sort(key=lambda x: -x["rank_delta"])
    result = {
        "methodology": (
            "overall_rank is a skill's position by posting frequency across all "
            "roles in the last 90 days; junior_rank is its position among "
            "experience_level='junior' postings only. Positive rank_delta means "
            "the skill matters more at entry level than the overall market ranking suggests."
        ),
        "shifts": shifts[:8],
        "sample_size_junior": sum(c for _, c in junior_rows),
    }
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/entry-level/universal-skills", summary="Skills present across the most role categories")
async def get_entry_level_universal_skills() -> dict:
    cache_key = "entry_level_universal_skills"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    from datetime import date, timedelta

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    jobs_t    = "jobs"       if is_sqlite else "market.jobs"
    skills_t  = "job_skills" if is_sqlite else "market.job_skills"
    since = str(date.today() - timedelta(days=90))

    with engine.connect() as conn:
        span_rows = conn.execute(text(f"""
            SELECT js.skill, COUNT(DISTINCT j.role_category) AS role_span, COUNT(*) AS total
            FROM {skills_t} js JOIN {jobs_t} j ON j.job_id = js.job_id
            WHERE j.scraped_at >= :since AND j.role_category IS NOT NULL
            GROUP BY js.skill HAVING COUNT(*) >= 5
            ORDER BY role_span DESC, total DESC LIMIT 8
        """), {"since": since}).fetchall()

        top_skills = [r[0] for r in span_rows]
        matrix: list[dict] = []
        if top_skills:
            from sqlalchemy import bindparam
            matrix_rows = conn.execute(
                text(f"""
                    SELECT js.skill, j.role_category, COUNT(*) AS cnt
                    FROM {skills_t} js JOIN {jobs_t} j ON j.job_id = js.job_id
                    WHERE j.scraped_at >= :since AND j.role_category IS NOT NULL
                      AND js.skill IN :skills
                    GROUP BY js.skill, j.role_category
                """).bindparams(bindparam("skills", expanding=True)),
                {"since": since, "skills": top_skills},
            ).fetchall()
            matrix = [{"skill": s, "role_category": rc, "count": c} for s, rc, c in matrix_rows]

    result = {
        "methodology": (
            "Top skills ranked by how many distinct role categories they appear "
            "in (last 90 days, roles with >=5 postings mentioning the skill), "
            "not just raw frequency — these are the skills that show up "
            "regardless of specialisation, the closest thing to a market-wide floor."
        ),
        "skills": [{"skill": s, "role_span": rs, "total": t} for s, rs, t in span_rows],
        "matrix": matrix,
    }
    cache.set(cache_key, result)
    return result


@app.get("/api/v1/market/entry-level/company-mix", summary="Company type mix, entry-level postings only")
async def get_entry_level_company_mix() -> dict:
    cache_key = "entry_level_company_mix"
    cached = cache.get(cache_key)
    if cached:
        return cached

    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text

    engine    = get_sync_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    table     = "jobs" if is_sqlite else "market.jobs"

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT company_stage, is_startup, COUNT(*) AS cnt
            FROM {table}
            WHERE experience_level = 'junior'
            GROUP BY company_stage, is_startup
        """)).fetchall()
        total = conn.execute(text(f"""
            SELECT COUNT(*) FROM {table} WHERE experience_level = 'junior'
        """)).scalar() or 0

    buckets: dict[str, int] = {
        "Scale-up (50–500)":   0,
        "Enterprise (500+)":   0,
        "Startup (<50)":       0,
        "Research / Academic": 0,
    }
    for stage, is_startup, cnt in rows:
        sl = (stage or "").lower()
        if any(k in sl for k in ("enterprise", "large", "corporate", "public")):
            buckets["Enterprise (500+)"] += cnt
        elif any(k in sl for k in ("research", "academic", "university", "institute")):
            buckets["Research / Academic"] += cnt
        elif any(k in sl for k in ("scale", "growth", "mid", "series b", "series c")):
            buckets["Scale-up (50–500)"] += cnt
        elif any(k in sl for k in ("startup", "early", "seed", "series a")) or is_startup:
            buckets["Startup (<50)"] += cnt
        else:
            buckets["Scale-up (50–500)"] += cnt

    result = {
        "sample_size": total,
        "mix": [
            {"type": t, "pct": round(n / max(total, 1) * 100, 1), "job_count": n}
            for t, n in buckets.items()
        ],
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
    from datetime import datetime, timezone

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
            freshness = round((datetime.now(timezone.utc) - last_run.replace(tzinfo=timezone.utc)).total_seconds() / 3600, 1)
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


@app.options("/api/v1/career/cv-analyse")
@app.options("/api/v1/career/analyse")
async def career_cors_preflight():
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
    )


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
    fastapi_response: Response,
    cv_file:     UploadFile = File(..., description="PDF or DOCX CV, max 5 MB"),
    target_role: str        = "ml_engineer",
    consent:     bool       = False,
) -> CVAnalysisReport:
    from marketforge.cv.scanner  import scan_file
    from marketforge.cv.parser   import parse_cv
    from marketforge.cv.ats_scorer import score_cv
    from marketforge.cv.gdpr     import build_gdpr_context, ConsentNotGiven

    ip = _get_client_ip(request)

    # ── Rate limit: 3 CV analyses per IP per hour (expensive operation) ────────
    if not limiter.is_allowed(f"cv_analyse:{ip}", limit=3, window_seconds=3600):
        raise HTTPException(status_code=429, detail="CV analysis rate limit exceeded (3/hour)")

    # ── Validate target_role against injection ────────────────────────────────
    role_check = validate_input(target_role, field_name="target_role", source_ip=ip, max_length=100)
    if not role_check.allowed:
        raise HTTPException(status_code=422, detail=role_check.rejection_reason)
    target_role = role_check.sanitised_text

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
    ats = await asyncio.to_thread(score_cv, cv, target_role)

    # ── Market match (SBERT) ───────────────────────────────────────────────────
    match_pct, _ = await asyncio.to_thread(_compute_market_match, ats.skills_found or [target_role], target_role)

    # ── Phase 2: ML gap analysis (demand × salary × recency priority scoring) ──
    from marketforge.cv.gap_analyser import analyse_gaps
    ml_gaps = await asyncio.to_thread(analyse_gaps, ats.skills_found, target_role, top_n=15)
    # ML-ranked missing skills (flat list for display, ordered by priority)
    skills_missing = [g.skill for g in ml_gaps.top_n(10)]
    # If DB has no market data yet fall back to simple heuristic
    if not skills_missing:
        skills_missing = [i["skill"] for i in _compute_skill_gaps(ats.skills_found, target_role)[:10]]

    # ── Phase 2: LLM gap plan seeded with ML-bucketed skills ──────────────────
    gap_plan, narrative = await _generate_cv_gap_plan(
        ats_score      = ats.total,
        skills_found   = ats.skills_found,
        ml_short_term  = [g.skill for g in ml_gaps.short_term[:4]],
        ml_mid_term    = [g.skill for g in ml_gaps.mid_term[:4]],
        ml_long_term   = [g.skill for g in ml_gaps.long_term[:3]],
        target_role    = target_role,
        match_pct      = match_pct,
    )

    # ── Output guardrails ──────────────────────────────────────────────────────
    from marketforge.agents.security.guardrails import validate_output
    narrative, _ = validate_output(narrative)

    logger.info(
        "cv.endpoint.complete",
        session=gdpr_ctx.session_token[:8],
        ats_score=ats.total,
        ats_grade=ats.grade,
        ml_gaps_total=len(ml_gaps.all_gaps),
    )

    fastapi_response.headers["Access-Control-Allow-Origin"] = "*"
    fastapi_response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    fastapi_response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

    return CVAnalysisReport(
        session_token     = gdpr_ctx.session_token,
        ats_score         = ats.total,
        ats_grade         = ats.grade,
        ats_breakdown     = CVATSBreakdown(**ats.breakdown),
        ats_issues        = ats.issues,
        skills_found      = ats.skills_found,
        skills_missing    = skills_missing,
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
    ml_short_term: list[str],
    ml_mid_term:   list[str],
    ml_long_term:  list[str],
    target_role:   str,
    match_pct:     float,
) -> tuple[CVGapPlan, str]:
    """
    LLM call to generate short/mid/long-term plan.
    Seeded with ML-ranked skill buckets from gap_analyser — receives structured data only,
    never raw CV text.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        found_str  = ", ".join(skills_found[:15]) or "none detected"
        short_str  = ", ".join(ml_short_term) or "none identified"
        mid_str    = ", ".join(ml_mid_term)   or "none identified"
        long_str   = ", ".join(ml_long_term)  or "none identified"

        prompt = f"""You are a UK AI/ML career advisor. Generate a structured career development plan.

STRUCTURED DATA (use only this — do not invent facts):
- ATS score: {ats_score:.0f}/100
- Target role: {target_role}
- Skills in CV: {found_str}
- Market match: {match_pct:.0f}%
- ML-ranked quick-win skills to add (0-3 months): {short_str}
- ML-ranked medium-effort skills (3-12 months): {mid_str}
- ML-ranked deep-expertise skills (12+ months): {long_str}

Respond in this exact format:

NARRATIVE: [2 sentences: current position assessment based on ATS score and market match]

SHORT_TERM (0-3 months):
- [specific action for each skill listed above, e.g. courses/certs]
- [action 2]
- [action 3]

MID_TERM (3-12 months):
- [project or bootcamp for each skill listed]
- [action 2]
- [action 3]

LONG_TERM (12+ months):
- [advanced specialisation or portfolio for each skill listed]
- [action 2]

Keep actions specific and achievable. Do not mention company names."""

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

        # Fill empty LLM buckets with ML-seed defaults
        def _seed(llm_items: list[str], ml_skills: list[str], verb: str) -> list[str]:
            if llm_items:
                return llm_items
            return [f"{verb} {s}" for s in ml_skills[:3]] if ml_skills else [f"{verb} top missing skills"]

        return (
            CVGapPlan(
                short_term = _seed(short_term, ml_short_term, "Complete a course or certification in"),
                mid_term   = _seed(mid_term,   ml_mid_term,   "Build a portfolio project using"),
                long_term  = _seed(long_term,  ml_long_term,  "Develop deep expertise in"),
            ),
            narrative,
        )

    except Exception as exc:
        logger.error("cv.gap_plan.error", error=str(exc))
        # Graceful degradation: return ML-bucketed skills directly as actionable items
        short_items = [f"Add {s} to your CV — quick course available" for s in ml_short_term[:3]] or ["Complete a course in top missing skills", "Add metrics to experience bullets", "Mirror job-ad keywords in CV"]
        mid_items   = [f"Build a project demonstrating {s}" for s in ml_mid_term[:3]]   or ["Build portfolio project using missing skills", "Complete relevant certification"]
        long_items  = [f"Develop deep expertise in {s}" for s in ml_long_term[:2]]      or ["Target senior roles after closing skill gaps"]
        return (
            CVGapPlan(short_term=short_items, mid_term=mid_items, long_term=long_items),
            f"CV scored {ats_score:.0f}/100 with a {match_pct:.0f}% market match for {target_role} roles. "
            f"Address the skill gaps above to improve ATS compatibility.",
        )


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics(request: Request) -> str:
    token = os.getenv("METRICS_TOKEN", "")
    if token:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            return PlainTextResponse("Forbidden", status_code=403)
    try:
        from prometheus_client import generate_latest
        return generate_latest().decode("utf-8")
    except Exception:
        return "# prometheus_client not available\n"
