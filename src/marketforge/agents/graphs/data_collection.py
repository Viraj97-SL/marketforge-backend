"""
MarketForge AI — Department 1: Data Collection LangGraph.

Graph topology:
  START
    └─ plan_collection          (reads DB state, decides active sources)
         └─ [Send fan-out] ──►  run_scraper  ×N  (one per active source, parallel)
                                     │
                                merge_results     (accumulated via state reducers)
                                     │
                               run_deduplication  (DeduplicationCoordinatorAgent)
                                     │
                                 write_to_db      (persists deduped jobs)
                                     │
                              reflect_collection  (quality gate, updates adaptive params)
                                     │
                                    END

LangSmith traces the full graph automatically when LANGCHAIN_TRACING_V2=true.
LangGraph checkpointing persists intermediate state so individual node failures
can be retried without re-running the entire fan-out.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from marketforge.agents.graphs.states import DataCollectionState
from marketforge.agents.data_collection.adzuna_agent import AdzunaDeepScoutAgent
from marketforge.agents.data_collection.reed_agent import ReedDeepScoutAgent
from marketforge.agents.data_collection.dedup_agent import DeduplicationCoordinatorAgent
from marketforge.agents.data_collection.additional_agents import (
    WellfoundDeepScoutAgent,
    ATSDirectDeepAgent,
    CareerPagesDeepCrawlerAgent,
)
from marketforge.agents.data_collection.funding_news_agent import FundingNewsDeepDiscoveryAgent
from marketforge.agents.data_collection.recruiter_agent import RecruiterBoardsDeepAgent
from marketforge.agents.data_collection.specialist_boards_agent import SpecialistBoardsDeepAgent
from marketforge.memory.postgres import AgentStateStore, JobStore, PipelineRunStore

logger = structlog.get_logger(__name__)

# ── Scraper agent factory ─────────────────────────────────────────────────────
# Each factory callable is invoked fresh per run to avoid shared state between
# parallel fan-out invocations.

_SCRAPER_FACTORY: dict[str, Any] = {
    "adzuna":            AdzunaDeepScoutAgent,
    "reed":              ReedDeepScoutAgent,
    "wellfound":         WellfoundDeepScoutAgent,
    "ats_direct":        ATSDirectDeepAgent,
    "career_pages":      CareerPagesDeepCrawlerAgent,
    "funding_news":      FundingNewsDeepDiscoveryAgent,
    "recruiter_boards":  RecruiterBoardsDeepAgent,
    "specialist_boards": SpecialistBoardsDeepAgent,
}

_ALL_SOURCES: list[str] = list(_SCRAPER_FACTORY.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — plan_collection
# ═══════════════════════════════════════════════════════════════════════════════

async def plan_collection(state: DataCollectionState) -> dict:
    """
    Read the 4-week yield history from PostgreSQL agent state.
    Decide which sources to activate and build the shared scraper context.
    Sources with 3+ consecutive zero-yield runs are skipped.
    """
    run_id = state.get("run_id") or str(uuid.uuid4())[:8]
    store  = AgentStateStore()
    db_state = store.load("data_collection_lead_v1", "data_collection")
    adaptive = db_state.get("adaptive_params", {})

    source_yields: dict[str, list[int]] = adaptive.get("source_yields", {})
    disabled: set[str] = set(adaptive.get("disabled_sources", []))

    active_sources: list[str] = []
    for src in _ALL_SOURCES:
        if src in disabled:
            logger.warning("data_collection.source_disabled", source=src)
            continue
        yields = source_yields.get(src, [])
        if len(yields) >= 3 and all(y == 0 for y in yields[-3:]):
            logger.warning("data_collection.source_degraded", source=src, recent_yields=yields[-3:])
            disabled.add(src)
            continue
        active_sources.append(src)

    scraper_context = {
        "run_id":   run_id,
        "watchlist": db_state.get("watchlist", []),
        "top_role_categories_by_demand": state.get("scraper_context", {}).get("top_role_categories", []),
    }

    logger.info(
        "data_collection.plan.done",
        run_id=run_id,
        active=active_sources,
        skipped=list(disabled),
    )
    return {
        "run_id":         run_id,
        "active_sources": active_sources,
        "scraper_context": scraper_context,
        "adaptive_params": adaptive,
        # Initialise accumulators so reducers have a base to merge into
        "raw_jobs":      [],
        "source_counts": {},
        "source_errors": {},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — run_scraper  (one invocation per source via Send fan-out)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_scraper(state: DataCollectionState) -> dict:
    """
    Execute a single scraper sub-agent.

    The state dict passed via Send contains 'source_name' plus the full
    shared scraper context.  Returns partial state updates that the
    Annotated reducers accumulate across all parallel invocations.
    """
    source_name = state.get("source_name", "")  # injected by Send
    if not source_name or source_name not in _SCRAPER_FACTORY:
        logger.error("data_collection.unknown_scraper", source=source_name)
        return {"raw_jobs": [], "source_counts": {source_name: 0}, "source_errors": {source_name: "unknown_source"}}

    agent_cls = _SCRAPER_FACTORY[source_name]
    agent = agent_cls()

    try:
        ctx    = state.get("scraper_context", {})
        result = await agent.run(ctx)
        jobs   = result.get("jobs", [])
        logger.info("data_collection.scraper.done", source=source_name, jobs=len(jobs))
        return {
            "raw_jobs":      jobs,
            "source_counts": {source_name: len(jobs)},
            "source_errors": {},
        }
    except Exception as exc:
        logger.error("data_collection.scraper.failed", source=source_name, error=str(exc))
        return {
            "raw_jobs":      [],
            "source_counts": {source_name: 0},
            "source_errors": {source_name: str(exc)},
        }


def dispatch_to_scrapers(state: DataCollectionState) -> list[Send]:
    """
    Router function: emits one Send per active source.
    LangGraph executes all Send targets concurrently (fan-out).
    """
    return [
        Send("run_scraper", {
            **state,
            "source_name": src,
        })
        for src in state.get("active_sources", [])
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — run_deduplication
# ═══════════════════════════════════════════════════════════════════════════════

async def run_deduplication(state: DataCollectionState) -> dict:
    """
    Run the DeduplicationCoordinatorAgent over the merged raw job list.
    Three-signal approach: exact hash, MinHash LSH, SBERT cosine.
    """
    agent = DeduplicationCoordinatorAgent()
    ctx   = {
        "raw_jobs": state.get("raw_jobs", []),
        "run_id":   state.get("run_id", ""),
    }
    result = await agent.run(ctx)

    deduped   = result.get("deduped_jobs", [])
    report    = result.get("dedup_report", {})
    raw_count = len(state.get("raw_jobs", []))

    logger.info(
        "data_collection.dedup.done",
        raw=raw_count,
        deduped=len(deduped),
        removed=raw_count - len(deduped),
    )
    return {
        "deduped_jobs": deduped,
        "dedup_report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — write_to_db
# ═══════════════════════════════════════════════════════════════════════════════

async def write_to_db(state: DataCollectionState) -> dict:
    """Persist all deduplicated jobs to market.jobs via JobStore."""
    job_store  = JobStore()
    run_id     = state.get("run_id", "")
    deduped    = state.get("deduped_jobs", [])
    write_errs = 0

    for job in deduped:
        try:
            job_store.upsert_job(job, run_id)
        except Exception as exc:
            write_errs += 1
            logger.warning("data_collection.write_error", job_id=getattr(job, "job_id", "?"), error=str(exc))

    if write_errs:
        logger.warning("data_collection.write_errors_total", count=write_errs, total=len(deduped))

    # Record pipeline run
    try:
        run_store = PipelineRunStore()
        run_store.finish(
            run_id,
            "running",
            jobs_scraped=len(state.get("raw_jobs", [])),
            jobs_new=len(deduped),
        )
    except Exception:
        pass

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — reflect_collection
# ═══════════════════════════════════════════════════════════════════════════════

async def reflect_collection(state: DataCollectionState) -> dict:
    """
    Quality gate:
    - Poor   → < 2 active sources returned results
    - Warning → < 30 deduped jobs total
    - Good   → everything looks healthy

    Updates adaptive_params (4-week yield history, disabled sources list)
    and persists them back to market.agent_state.
    """
    adaptive      = state.get("adaptive_params", {})
    source_counts = state.get("source_counts", {})
    deduped       = state.get("deduped_jobs", [])

    source_yields: dict[str, list[int]] = adaptive.get("source_yields", {})
    for src, count in source_counts.items():
        hist = source_yields.get(src, [])
        hist.append(count)
        source_yields[src] = hist[-4:]  # keep last 4 runs

    # Auto-disable sources with 3+ zero-yield runs
    disabled = list(adaptive.get("disabled_sources", []))
    for src, hist in source_yields.items():
        if len(hist) >= 3 and all(y == 0 for y in hist[-3:]) and src not in disabled:
            disabled.append(src)
            logger.warning("data_collection.auto_disabled", source=src)

    adaptive["source_yields"]    = source_yields
    adaptive["disabled_sources"] = disabled

    n_active = len([c for c in source_counts.values() if c > 0])
    total    = len(deduped)
    quality  = "good" if n_active >= 2 and total >= 30 else ("warning" if n_active >= 2 else "poor")

    logger.info(
        "data_collection.reflect.done",
        quality=quality,
        deduped=total,
        active_sources=n_active,
    )

    # Persist updated adaptive params
    try:
        store    = AgentStateStore()
        db_state = store.load("data_collection_lead_v1", "data_collection")
        db_state["adaptive_params"] = adaptive
        db_state["last_yield"]      = total
        db_state["last_run_at"]     = datetime.utcnow().isoformat()
        db_state["run_count"]       = db_state.get("run_count", 0) + 1
        store.save(db_state)
    except Exception as exc:
        logger.warning("data_collection.state_persist_failed", error=str(exc))

    return {
        "collection_quality": quality,
        "adaptive_params":    adaptive,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_data_collection_graph() -> StateGraph:
    """
    Construct the Department 1 StateGraph.

    Returns the uncompiled graph so callers can attach their own checkpointer
    (MemorySaver for tests, AsyncPostgresSaver for production).
    """
    graph = StateGraph(DataCollectionState)

    # Register nodes
    graph.add_node("plan_collection",    plan_collection)
    graph.add_node("run_scraper",        run_scraper)
    graph.add_node("run_deduplication",  run_deduplication)
    graph.add_node("write_to_db",        write_to_db)
    graph.add_node("reflect_collection", reflect_collection)

    # plan → fan-out to scrapers (dynamic, via Send)
    graph.add_edge(START, "plan_collection")
    graph.add_conditional_edges(
        "plan_collection",
        dispatch_to_scrapers,
        ["run_scraper"],
    )

    # All scraper fan-out results converge at deduplication
    graph.add_edge("run_scraper",        "run_deduplication")
    graph.add_edge("run_deduplication",  "write_to_db")
    graph.add_edge("write_to_db",        "reflect_collection")
    graph.add_edge("reflect_collection", END)

    return graph


# Default compiled graph (MemorySaver — swap for AsyncPostgresSaver in prod)
_checkpointer = MemorySaver()
data_collection_graph = build_data_collection_graph().compile(
    checkpointer=_checkpointer,
    name="data_collection",
)


# ── Entry point for Airflow / API ────────────────────────────────────────────

async def run_data_collection_pipeline(
    run_id: str | None = None,
    top_role_categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Invoke the compiled Data Collection graph.

    This is the function called by the Airflow `scrape_all_sources` task.
    Returns a summary dict suitable for XCom.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": run_id}}

    initial_state: DataCollectionState = {
        "run_id":         run_id,
        "scraper_context": {"top_role_categories": top_role_categories or []},
        "raw_jobs":       [],
        "source_counts":  {},
        "source_errors":  {},
    }

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_data_collection_graph().compile(
            checkpointer=checkpointer,
            name="data_collection",
        )
        final_state = await graph.ainvoke(initial_state, config=config)

    return {
        "run_id":        final_state.get("run_id", run_id),
        "jobs_raw":      len(final_state.get("raw_jobs", [])),
        "jobs_new":      len(final_state.get("deduped_jobs", [])),
        "source_counts": final_state.get("source_counts", {}),
        "source_errors": final_state.get("source_errors", {}),
        "quality":       final_state.get("collection_quality", "unknown"),
    }
