"""
MarketForge AI — Department 3: Market Analysis LangGraph.

Graph topology (all 7 sub-agents run in parallel after START):

  START
    ├─► skill_demand_node   ─┐
    ├─► salary_intel_node   ─┤
    ├─► sponsorship_node    ─┤
    ├─► velocity_node       ─┼─► compile_snapshot ─► END
    ├─► cooccurrence_node   ─┤
    ├─► geo_dist_node       ─┤
    └─► techstack_node      ─┘

LangGraph executes all 7 nodes concurrently (superstep fan-in).
compile_snapshot assembles the MarketSnapshot and persists it to PostgreSQL.
Zero LLM calls — all computation is SQL + pandas.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import MarketAnalysisState
from marketforge.agents.market_analysis.lead_agent import (
    SkillDemandAnalystAgent,
    SalaryIntelligenceAgent,
    SponsorshipTrackerAgent,
    HiringVelocityAgent,
    SkillCoOccurrenceAgent,
    GeographicDistributionAgent,
    TechStackFingerprintAgent,
)

logger = structlog.get_logger(__name__)


def _default_week() -> str:
    today = date.today()
    return str(today - timedelta(days=today.weekday()))


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel analysis nodes — each wraps one DeepAgent sub-agent
# ═══════════════════════════════════════════════════════════════════════════════

async def skill_demand_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — SkillDemandAnalystAgent."""
    agent  = SkillDemandAnalystAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"skill_trends": result}


async def salary_intel_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — SalaryIntelligenceAgent."""
    agent  = SalaryIntelligenceAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"salary_stats": result}


async def sponsorship_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — SponsorshipTrackerAgent."""
    agent  = SponsorshipTrackerAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"sponsorship_data": result}


async def velocity_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — HiringVelocityAgent."""
    agent  = HiringVelocityAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"velocity_data": result}


async def cooccurrence_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — SkillCoOccurrenceAgent."""
    agent  = SkillCoOccurrenceAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"cooccurrence_pairs": result.get("top_skill_pairs", [])}


async def geo_dist_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — GeographicDistributionAgent."""
    agent  = GeographicDistributionAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"geo_distribution": result}


async def techstack_node(state: MarketAnalysisState) -> dict:
    """LangGraph node — TechStackFingerprintAgent."""
    agent  = TechStackFingerprintAgent()
    result = await agent.run({"week_start": state.get("week_start", _default_week())})
    return {"tech_archetypes": result.get("archetypes", [])}


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-in node — compile_snapshot
# ═══════════════════════════════════════════════════════════════════════════════

async def compile_snapshot(state: MarketAnalysisState) -> dict:
    """
    Assembles and persists the weekly MarketSnapshot from all parallel outputs.
    Runs after ALL 7 analysis nodes have completed (LangGraph fan-in).
    """
    from datetime import datetime
    from marketforge.memory.postgres import get_sync_engine
    from sqlalchemy import text
    import json

    week = state.get("week_start", _default_week())

    skill_trends     = state.get("skill_trends",     {})
    salary_stats     = state.get("salary_stats",     {})
    sponsorship_data = state.get("sponsorship_data", {})
    velocity_data    = state.get("velocity_data",    {})
    geo_dist         = state.get("geo_distribution", {})
    archetypes       = state.get("tech_archetypes",  [])

    snapshot: dict[str, Any] = {
        "week_start":        week,
        "job_count":         velocity_data.get("total_jobs_this_week", 0),
        "top_skills":        skill_trends.get("top_skills", {}),
        "rising_skills":     skill_trends.get("rising_skills", []),
        "declining_skills":  skill_trends.get("declining_skills", []),
        "salary_p10":        salary_stats.get("p10"),
        "salary_p25":        salary_stats.get("p25"),
        "salary_p50":        salary_stats.get("p50"),
        "salary_p75":        salary_stats.get("p75"),
        "salary_p90":        salary_stats.get("p90"),
        "salary_sample_size": salary_stats.get("sample_size", 0),
        "sponsorship_rate":  sponsorship_data.get("sponsorship_rate", 0),
        "remote_rate":       geo_dist.get("remote_rate", 0),
        "geo_breakdown":     geo_dist.get("city_breakdown", {}),
        "tech_archetypes":   archetypes,
        "generated_at":      datetime.utcnow().isoformat(),
    }

    # Persist to market.weekly_snapshots
    try:
        engine   = get_sync_engine()
        snaps_t  = "weekly_snapshots" if engine.dialect.name == "sqlite" else "market.weekly_snapshots"
        top_json = json.dumps(snapshot.get("top_skills", {}))

        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO {snaps_t}
                    (week_start, role_category, job_count, top_skills,
                     salary_p25, salary_p50, salary_p75, salary_sample_size,
                     sponsorship_rate, remote_rate)
                VALUES (:ws, 'all', :jc, :ts, :p25, :p50, :p75, :sn, :sr, :rr)
                ON CONFLICT (week_start, role_category) DO UPDATE SET
                    job_count        = EXCLUDED.job_count,
                    top_skills       = EXCLUDED.top_skills,
                    salary_p25       = EXCLUDED.salary_p25,
                    salary_p50       = EXCLUDED.salary_p50,
                    salary_p75       = EXCLUDED.salary_p75,
                    salary_sample_size = EXCLUDED.salary_sample_size,
                    sponsorship_rate = EXCLUDED.sponsorship_rate,
                    remote_rate      = EXCLUDED.remote_rate
            """), {
                "ws":  week,
                "jc":  snapshot["job_count"],
                "ts":  top_json,
                "p25": snapshot.get("salary_p25"),
                "p50": snapshot.get("salary_p50"),
                "p75": snapshot.get("salary_p75"),
                "sn":  snapshot.get("salary_sample_size", 0),
                "sr":  snapshot.get("sponsorship_rate", 0),
                "rr":  snapshot.get("remote_rate", 0),
            })
            conn.commit()
        logger.info("market_analysis.snapshot.saved", week=week, jobs=snapshot["job_count"])
    except Exception as exc:
        logger.error("market_analysis.snapshot.save_failed", error=str(exc))

    return {
        "weekly_snapshot":  snapshot,
        "analysis_quality": "good" if snapshot["job_count"] > 0 else "warning",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

_ANALYSIS_NODES: list[tuple[str, Any]] = [
    ("skill_demand",  skill_demand_node),
    ("salary_intel",  salary_intel_node),
    ("sponsorship",   sponsorship_node),
    ("velocity",      velocity_node),
    ("cooccurrence",  cooccurrence_node),
    ("geo_dist",      geo_dist_node),
    ("techstack",     techstack_node),
]


def build_market_analysis_graph() -> StateGraph:
    """
    Construct the Department 3 StateGraph.

    All 7 sub-agent nodes are wired to run in parallel from START.
    compile_snapshot fans-in and runs only after all 7 complete.
    """
    graph = StateGraph(MarketAnalysisState)

    # Register all parallel sub-agent nodes
    for node_name, node_fn in _ANALYSIS_NODES:
        graph.add_node(node_name, node_fn)

    # compile_snapshot runs after all analysis nodes complete
    graph.add_node("compile_snapshot", compile_snapshot)

    # Wire: START → all analysis nodes in parallel
    for node_name, _ in _ANALYSIS_NODES:
        graph.add_edge(START, node_name)

    # Wire: all analysis nodes → compile_snapshot (fan-in)
    for node_name, _ in _ANALYSIS_NODES:
        graph.add_edge(node_name, "compile_snapshot")

    graph.add_edge("compile_snapshot", END)

    return graph


_checkpointer = MemorySaver()
market_analysis_graph = build_market_analysis_graph().compile(
    checkpointer=_checkpointer,
    name="market_analysis",
)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_market_analysis_pipeline(
    run_id: str | None = None,
    week_start: str | None = None,
) -> dict[str, Any]:
    """Called by the Airflow aggregate_market_stats task."""
    import uuid
    run_id     = run_id or str(uuid.uuid4())[:8]
    week_start = week_start or _default_week()
    config     = {"configurable": {"thread_id": run_id}}

    initial_state: MarketAnalysisState = {
        "run_id":     run_id,
        "week_start": week_start,
    }

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_market_analysis_graph().compile(
            checkpointer=checkpointer,
            name="market_analysis",
        )
        final = await graph.ainvoke(initial_state, config=config)

    return {
        "run_id":          run_id,
        "week_start":      week_start,
        "snapshot_jobs":   final.get("weekly_snapshot", {}).get("job_count", 0),
        "analysis_quality": final.get("analysis_quality", "unknown"),
    }
