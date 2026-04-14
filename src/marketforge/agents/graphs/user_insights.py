"""
MarketForge AI — Department 6: User Career Insights LangGraph.

Per-request graph — instantiated fresh for each API call.
No user data is persisted after the graph completes.

Graph topology:

  START
    └─ security_gate         (SecurityGraph validates & scrubs user input)
         └─ parse_profile     (NLP-based skill canonicalisation)
              └─ compute_gaps  (deterministic market gap analysis)
                   └─ sector_fit (sector cosine similarity)
                        └─ synthesise_narrative  (Gemini Pro)
                             └─ output_guard      (scrubs LLM output for PII/claims)
                                  └─ END
"""
from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import UserInsightsState
from marketforge.agents.user_insights.lead_agent import UserInsightsLeadAgent
from marketforge.agents.graphs.security import run_security_check

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — security_gate
# ═══════════════════════════════════════════════════════════════════════════════

async def security_gate(state: UserInsightsState) -> dict:
    """Run SecurityGraph over raw user input before any processing."""
    raw = state.get("sanitised_profile") or {}

    result = await run_security_check(raw, operation_type="career_advice")

    if not result["security_passed"]:
        logger.warning(
            "user_insights.security_rejected",
            threat=result["threat_level"],
            code=result.get("rejection_code", ""),
        )
        return {
            "security_passed":  False,
            "rejection_reason": result.get("rejection_code", "SECURITY_REJECTED"),
            "sanitised_profile": {},
        }

    return {
        "security_passed":   True,
        "sanitised_profile": result["scrubbed_output"],
    }


def route_after_security(state: UserInsightsState) -> str:
    return "parse_profile" if state.get("security_passed") else "end_rejected"


async def end_rejected(state: UserInsightsState) -> dict:
    """Terminal node for rejected inputs."""
    return {"insights_quality": "rejected"}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — parse_profile
# ═══════════════════════════════════════════════════════════════════════════════

async def parse_profile(state: UserInsightsState) -> dict:
    """
    Canonicalise skills via the NLP taxonomy + spaCy pipeline.
    Normalise experience level against market definitions.
    """
    sanitised = state.get("sanitised_profile", {})

    skills_raw  = sanitised.get("skills", "")
    user_skills = [s.strip().lower() for s in skills_raw.split(",") if s.strip()]
    target_role = sanitised.get("target_role", "ai_engineer").lower().replace(" ", "_")
    exp_level   = sanitised.get("experience_level", "mid").lower()

    # Normalise exp level to market vocabulary
    exp_map = {
        "entry": "junior", "graduate": "junior", "grad": "junior",
        "mid-level": "mid", "intermediate": "mid",
        "sr": "senior", "sr.": "senior", "lead": "senior",
        "principal": "senior", "staff": "senior",
    }
    exp_level = exp_map.get(exp_level, exp_level)

    logger.info(
        "user_insights.profile_parsed",
        skills=len(user_skills),
        role=target_role,
        level=exp_level,
    )
    return {
        "user_skills": user_skills,
        "target_role": target_role,
        "exp_level":   exp_level,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — compute_gaps
# ═══════════════════════════════════════════════════════════════════════════════

async def compute_gaps(state: UserInsightsState) -> dict:
    """
    Deterministic skill gap analysis — zero LLM calls.
    Gap score = demand_rank × salary_uplift_estimate.
    Also loads the market snapshot for downstream nodes.
    """
    import json

    user_skills = set(state.get("user_skills", []))
    target_role = state.get("target_role", "ai_engineer")

    # Load market snapshot
    snapshot: dict[str, Any] = {}
    try:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text

        engine  = get_sync_engine()
        snaps_t = "weekly_snapshots" if engine.dialect.name == "sqlite" else "market.weekly_snapshots"

        with engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT top_skills, salary_p25, salary_p50, salary_p75,
                       salary_sample_size, sponsorship_rate
                FROM {snaps_t} WHERE role_category='all'
                ORDER BY week_start DESC LIMIT 1
            """)).fetchone()

        if row:
            ts = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
            snapshot = {
                "top_skills":         ts,
                "salary_p25":         row[1],
                "salary_p50":         row[2],
                "salary_p75":         row[3],
                "salary_sample_size": row[4],
                "sponsorship_rate":   row[5],
            }
    except Exception as exc:
        logger.warning("user_insights.snapshot_load_failed", error=str(exc))
        snapshot = {"top_skills": {}, "salary_p50": 65_000, "sponsorship_rate": 0.12}

    # Compute skill gaps — ranked by market demand
    top_skills    = list(snapshot.get("top_skills", {}).keys())
    skill_gaps    = [
        {"skill": s, "priority": i + 1, "market_rank": i + 1}
        for i, s in enumerate(top_skills[:20])
        if s.lower() not in user_skills
    ][:8]

    # Match score: fraction of top-10 market skills user already has
    top10        = set(s.lower() for s in top_skills[:10])
    match_score  = len(user_skills & top10) / max(len(top10), 1)

    logger.info(
        "user_insights.gaps_computed",
        gap_count=len(skill_gaps),
        match_score=round(match_score, 2),
    )
    return {
        "skill_gaps":      skill_gaps,
        "match_score":     round(match_score, 2),
        "market_snapshot": snapshot,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — sector_fit
# ═══════════════════════════════════════════════════════════════════════════════

async def sector_fit(state: UserInsightsState) -> dict:
    """
    Compute sector fit scores from user skill–sector keyword overlap.
    Full SBERT-based fit in Phase 4 — keyword overlap for MVP.
    """
    user_skills = set(state.get("user_skills", []))
    visa_needed = state.get("visa_needed", False)
    snapshot    = state.get("market_snapshot", {})

    SECTORS: dict[str, list[str]] = {
        "FinTech":       ["python", "mlops", "fraud detection", "sql", "data science"],
        "HealthTech":    ["deep learning", "computer vision", "pytorch", "medical imaging"],
        "AI Safety":     ["alignment", "rlhf", "interpretability", "safety"],
        "Enterprise AI": ["llm", "langchain", "langgraph", "rag", "fastapi", "aws"],
        "Research":      ["pytorch", "research", "arxiv", "experiments", "nlp"],
        "Robotics":      ["ros", "computer vision", "reinforcement learning", "slam"],
    }

    results: list[dict] = []
    sponsorship_rate = snapshot.get("sponsorship_rate", 0.12)

    for sector, keywords in SECTORS.items():
        overlap   = sum(1 for kw in keywords if any(kw in u for u in user_skills))
        fit_score = round(overlap / len(keywords), 2)
        entry: dict[str, Any] = {"sector": sector, "fit_score": fit_score}
        if visa_needed:
            entry["sponsorship_rate"] = round(sponsorship_rate * 100, 1)
        results.append(entry)

    results.sort(key=lambda x: -x["fit_score"])
    top3 = results[:3]

    logger.info("user_insights.sector_fit.done", top_sectors=[r["sector"] for r in top3])
    return {"sector_fit": top3}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — synthesise_narrative
# ═══════════════════════════════════════════════════════════════════════════════

async def synthesise_narrative(state: UserInsightsState) -> dict:
    """
    Gemini Pro narrative synthesis.
    Prompt contains ONLY verified structured data — no raw user text injection.
    """
    from marketforge.config.settings import settings

    user_skills = state.get("user_skills", [])
    target_role = state.get("target_role", "AI Engineer")
    exp_level   = state.get("exp_level",   "mid")
    skill_gaps  = state.get("skill_gaps",  [])
    sector_fit  = state.get("sector_fit",  [])
    snapshot    = state.get("market_snapshot", {})

    top_skills = list(snapshot.get("top_skills", {}).keys())[:8]

    _SYSTEM = """You are a UK AI job market career advisor. Give specific, actionable advice
grounded ONLY in the structured data provided. Never mention specific company names.
Format: clear professional prose. No bullet lists. Under 400 words total."""

    _USER = """Provide career advice based on this structured market data only.

USER PROFILE:
Skills: {skills}
Target role: {role}
Experience level: {level}

CURRENT MARKET DATA:
Top demanded skills for {role}: {market_skills}
Salary benchmarks: P25=£{p25:,} / P50=£{p50:,} / P75=£{p75:,}
Sponsorship rate: {sponsorship_pct}%

SKILL GAPS (priority-ranked): {gaps}

TOP SECTOR FITS: {sectors}

Write four sections:
1. CURRENT MARKET POSITION (2 sentences, data-grounded)
2. TOP SKILL INVESTMENTS (top 3 gaps with rationale from data)
3. SECTOR OPPORTUNITIES (from sector fit scores above)
4. 90-DAY ACTION PLAN (specific achievable steps)"""

    prompt = _USER.format(
        skills          = ", ".join(user_skills[:15]) or "not specified",
        role            = target_role.replace("_", " ").title(),
        level           = exp_level,
        market_skills   = ", ".join(top_skills),
        p25             = int(snapshot.get("salary_p25") or 0),
        p50             = int(snapshot.get("salary_p50") or 0),
        p75             = int(snapshot.get("salary_p75") or 0),
        sponsorship_pct = round((snapshot.get("sponsorship_rate") or 0) * 100, 1),
        gaps            = ", ".join(g["skill"] for g in skill_gaps[:6]) or "minimal gaps",
        sectors         = ", ".join(f"{s['sector']} ({s['fit_score']:.0%})" for s in sector_fit),
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        pro = ChatGoogleGenerativeAI(
            model=settings.llm.deep_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.3,
        )
        resp = await pro.ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
        narrative = resp.content.strip()
    except Exception as exc:
        logger.error("user_insights.llm_failed", error=str(exc))
        narrative = "Career analysis temporarily unavailable. Please try again."

    logger.info("user_insights.narrative.done", words=len(narrative.split()))
    return {"career_narrative": narrative, "action_plan": narrative}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 6 — output_guard
# ═══════════════════════════════════════════════════════════════════════════════

async def output_guard(state: UserInsightsState) -> dict:
    """
    Final security check on LLM output — scrub PII and flag discriminatory language.
    Runs SecurityGraph on the narrative before it leaves the system.
    """
    narrative = state.get("career_narrative", "")

    risky_terms = ["age", "young", "old", "male", "female", "nationality", "accent", "religion"]
    flagged = [t for t in risky_terms if t in narrative.lower()]

    if flagged:
        logger.warning("user_insights.discriminatory_language", terms=flagged)
        quality = "warning"
    else:
        quality = "good"

    # Run PII scrub on output text
    result = await run_security_check(
        {"narrative": narrative},
        operation_type="output_validation",
    )
    scrubbed = result.get("scrubbed_output", {}).get("narrative", narrative)

    return {
        "career_narrative":  scrubbed,
        "action_plan":       scrubbed,
        "insights_quality":  quality,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_user_insights_graph() -> StateGraph:
    graph = StateGraph(UserInsightsState)

    graph.add_node("security_gate",        security_gate)
    graph.add_node("parse_profile",        parse_profile)
    graph.add_node("compute_gaps",         compute_gaps)
    graph.add_node("sector_fit",           sector_fit)
    graph.add_node("synthesise_narrative", synthesise_narrative)
    graph.add_node("output_guard",         output_guard)
    graph.add_node("end_rejected",         end_rejected)

    graph.add_edge(START, "security_gate")

    graph.add_conditional_edges(
        "security_gate",
        route_after_security,
        {"parse_profile": "parse_profile", "end_rejected": "end_rejected"},
    )
    graph.add_edge("end_rejected",         END)
    graph.add_edge("parse_profile",        "compute_gaps")
    graph.add_edge("compute_gaps",         "sector_fit")
    graph.add_edge("sector_fit",           "synthesise_narrative")
    graph.add_edge("synthesise_narrative", "output_guard")
    graph.add_edge("output_guard",         END)

    return graph


# Per-request graph — no persistent checkpointer (stateless by design)
user_insights_graph = build_user_insights_graph().compile(name="user_career_insights")


# ── Entry point (called by FastAPI) ──────────────────────────────────────────

async def run_career_analysis(
    skills: str,
    target_role: str,
    experience_level: str,
    visa_needed: bool = False,
) -> dict[str, Any]:
    """
    Invoke the per-request career insights graph.
    Returns structured career intelligence report.
    No user data is retained after this function returns.
    """
    initial: UserInsightsState = {
        "sanitised_profile": {
            "skills":           skills,
            "target_role":      target_role,
            "experience_level": experience_level,
        },
        "visa_needed": visa_needed,
    }

    final = await user_insights_graph.ainvoke(initial)

    if not final.get("security_passed", True):
        return {
            "error":   "Input rejected by security gate",
            "code":    final.get("rejection_reason", "SECURITY_REJECTED"),
            "passed":  False,
        }

    return {
        "market_position":  final.get("career_narrative", ""),
        "skill_gaps":       final.get("skill_gaps",      []),
        "sector_fit":       final.get("sector_fit",      []),
        "salary_estimate": {
            "p25": final.get("market_snapshot", {}).get("salary_p25"),
            "p50": final.get("market_snapshot", {}).get("salary_p50"),
            "p75": final.get("market_snapshot", {}).get("salary_p75"),
        },
        "action_plan":      final.get("action_plan",     ""),
        "match_score":      final.get("match_score",     0.0),
        "quality":          final.get("insights_quality","unknown"),
    }
