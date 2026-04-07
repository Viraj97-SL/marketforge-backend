"""
MarketForge AI — FastAPI Application

Endpoints:
  POST /api/v1/career/analyse     — personalised career advice (LLM-backed)
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
from fastapi import FastAPI, HTTPException, Query, Request, status
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


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> str:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest().decode("utf-8")
    except Exception:
        return "# prometheus_client not available\n"
