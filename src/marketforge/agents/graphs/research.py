"""
MarketForge AI — Department 4: Research Intelligence LangGraph.

Graph topology:

  START
    └─ load_research_context    (reads top skills from DB for query building)
         ├─► arxiv_monitor_node       (fetches & summarises arXiv papers)
         └─► emerging_signal_node     (adoption lag analysis)
              │
         merge_research_signals       (aggregates + persists signals)
              │
             END

arxiv_monitor and emerging_signal run in parallel.
"""
from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import ResearchState
from marketforge.agents.research.lead_agent import (
    arXivMonitorAgent,
    EmergingTechSignalAgent,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — load_research_context
# ═══════════════════════════════════════════════════════════════════════════════

async def load_research_context(state: ResearchState) -> dict:
    """
    Pull the current week's top skills from market.weekly_snapshots.
    These drive the arXiv query matrix (trending skill → targeted paper search).
    """
    import json
    top_skills: dict[str, int] = {}

    try:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text

        engine  = get_sync_engine()
        snaps_t = "weekly_snapshots" if engine.dialect.name == "sqlite" else "market.weekly_snapshots"

        with engine.connect() as conn:
            row = conn.execute(text(
                f"SELECT top_skills FROM {snaps_t} ORDER BY week_start DESC LIMIT 1"
            )).fetchone()

        if row and row[0]:
            top_skills = json.loads(row[0]) if isinstance(row[0], str) else row[0]

        logger.info("research.context_loaded", skills_count=len(top_skills))
    except Exception as exc:
        logger.warning("research.context_load_failed", error=str(exc))

    return {"top_skills": top_skills}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — arxiv_monitor_node  (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

async def arxiv_monitor_node(state: ResearchState) -> dict:
    """LangGraph node — arXivMonitorAgent."""
    agent  = arXivMonitorAgent()
    result = await agent.run({"top_skills": state.get("top_skills", {})})
    return {
        "research_papers": result.get("research_papers", []),
        "summary_cards":   result.get("summary_cards",   []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — emerging_signal_node  (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

async def emerging_signal_node(state: ResearchState) -> dict:
    """LangGraph node — EmergingTechSignalAgent."""
    agent  = EmergingTechSignalAgent()
    result = await agent.run({"top_skills": state.get("top_skills", {})})
    return {
        "emerging_signals":       result.get("emerging_signals", []),
        "confirmed_adoptions":    result.get("confirmed_adoptions", []),
        "mean_adoption_lag_days": result.get("mean_adoption_lag_days", 42.0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — merge_research_signals  (fan-in)
# ═══════════════════════════════════════════════════════════════════════════════

async def merge_research_signals(state: ResearchState) -> dict:
    """
    Aggregate outputs from both parallel research sub-agents.
    Runs after BOTH arxiv_monitor and emerging_signal complete.
    """
    papers   = state.get("research_papers", [])
    signals  = state.get("emerging_signals", [])
    adoptions= state.get("confirmed_adoptions", [])
    lag      = state.get("mean_adoption_lag_days", 42.0)

    quality = "good" if (papers or signals) else "warning"

    logger.info(
        "research.merge.done",
        papers=len(papers),
        signals=len(signals),
        confirmed_adoptions=len(adoptions),
        mean_lag_days=lag,
        quality=quality,
    )
    return {"research_quality": quality}


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("load_research_context", load_research_context)
    graph.add_node("arxiv_monitor",         arxiv_monitor_node)
    graph.add_node("emerging_signal",       emerging_signal_node)
    graph.add_node("merge_research_signals",merge_research_signals)

    # Entry
    graph.add_edge(START, "load_research_context")

    # Parallel fan-out after context load
    graph.add_edge("load_research_context", "arxiv_monitor")
    graph.add_edge("load_research_context", "emerging_signal")

    # Fan-in to merge
    graph.add_edge("arxiv_monitor",   "merge_research_signals")
    graph.add_edge("emerging_signal", "merge_research_signals")

    graph.add_edge("merge_research_signals", END)

    return graph


_checkpointer = MemorySaver()
research_graph = build_research_graph().compile(
    checkpointer=_checkpointer,
    name="research_intelligence",
)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_research_pipeline(run_id: str | None = None) -> dict[str, Any]:
    """Called by the Airflow research intelligence task."""
    import uuid
    run_id = run_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": run_id}}

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_research_graph().compile(
            checkpointer=checkpointer,
            name="research_intelligence",
        )
        final = await graph.ainvoke({"run_id": run_id}, config=config)

    return {
        "run_id":        run_id,
        "papers":        len(final.get("research_papers", [])),
        "signals":       len(final.get("emerging_signals", [])),
        "adoptions":     len(final.get("confirmed_adoptions", [])),
        "quality":       final.get("research_quality", "unknown"),
    }
