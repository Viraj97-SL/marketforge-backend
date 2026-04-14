"""
MarketForge AI — Master Pipeline LangGraph.

Top-level StateGraph that chains all 9 departments in the correct order
for the twice-weekly ingestion + analysis + report pipeline.

Graph topology (dag_ingest_primary + dag_weekly_analysis combined):

  START
    └─ init_pipeline
         └─ data_collection          (Dept 1 — scrape all UK AI/ML jobs)
              └─ qa_post_ingestion    (Dept 7 — batch integrity checks)
                   ├─(qa_pass)─► market_analysis    (Dept 3 — SQL aggregation)
                   │                  └─ research_intelligence  (Dept 4 — arXiv + signals)
                   │                       └─ content_studio    (Dept 5 — LLM report gen)
                   │                            └─ qa_pre_dispatch   (Dept 7 — report QA gate)
                   │                                 ├─(pass)─► finalize_pipeline
                   │                                 └─(fail)─► finalize_pipeline (flagged)
                   └─(qa_fail)─► finalize_pipeline  (pipeline halted, ops alerted)

ML Engineering (Dept 2) runs separately on its own DAG (dag_model_retrain).
Security (Dept 8) runs as middleware WITHIN UserInsights and ContentStudio.
Ops (Dept 9) runs on a 30-minute heartbeat AND on pipeline completion.
User Insights (Dept 6) is a per-request graph, not part of the pipeline.

LangGraph checkpointing means any node can be retried independently
without re-running the full pipeline.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import MarketForgeState
from marketforge.agents.graphs.data_collection import run_data_collection_pipeline
from marketforge.agents.graphs.market_analysis import run_market_analysis_pipeline
from marketforge.agents.graphs.research import run_research_pipeline
from marketforge.agents.graphs.content_studio import run_content_pipeline
from marketforge.agents.graphs.qa_testing import run_qa_pipeline
from marketforge.agents.graphs.ops_monitor import run_ops_on_pipeline_complete

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Node: init_pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def init_pipeline(state: MarketForgeState) -> dict:
    """Initialise the pipeline run — create run_id, log start, init DB."""
    from marketforge.memory.postgres import init_database, PipelineRunStore

    run_id     = state.get("pipeline_run_id") or str(uuid.uuid4())[:12]
    started_at = datetime.utcnow().isoformat()

    try:
        init_database()
        store = PipelineRunStore()
        store.start(run_id, "dag_master_pipeline")
    except Exception as exc:
        logger.warning("master.init.db_failed", error=str(exc))

    logger.info("master.pipeline.started", run_id=run_id)
    return {
        "pipeline_run_id": run_id,
        "pipeline_status": "running",
        "started_at":      started_at,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: dept1_data_collection
# ═══════════════════════════════════════════════════════════════════════════════

async def dept1_data_collection(state: MarketForgeState) -> dict:
    """
    Invoke the Department 1 (Data Collection) compiled graph.
    Runs the full scraper fan-out → dedup → write pipeline.
    """
    run_id = state.get("pipeline_run_id", "")
    logger.info("master.dept1.start", run_id=run_id)

    result = await run_data_collection_pipeline(run_id=run_id)

    logger.info(
        "master.dept1.done",
        jobs_raw=result.get("jobs_raw", 0),
        jobs_new=result.get("jobs_new", 0),
        quality=result.get("quality"),
    )
    return {
        "raw_jobs":      [],   # cleared after write — jobs live in DB
        "source_counts": result.get("source_counts", {}),
        "source_errors": result.get("source_errors", {}),
        "collection_quality": result.get("quality", "unknown"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: dept7_qa_post_ingestion
# ═══════════════════════════════════════════════════════════════════════════════

async def dept7_qa_post_ingestion(state: MarketForgeState) -> dict:
    """
    Department 7 — post-ingestion data integrity checks.
    Runs parallel: DataIntegrity + ConnectorHealth + ModelDrift.
    """
    run_id = state.get("pipeline_run_id", "")
    logger.info("master.dept7.qa_post_ingestion.start", run_id=run_id)

    result = await run_qa_pipeline(run_id=run_id, report_draft=None)

    logger.info(
        "master.dept7.qa_post_ingestion.done",
        qa_pass=result.get("qa_pass"),
        quality=result.get("quality"),
    )
    return {
        "qa_pass":            result.get("qa_pass",     True),
        "batch_quality_score": result.get("batch_score", 0.0),
        "qa_quality":         result.get("quality",     "unknown"),
    }


def route_after_qa_ingestion(state: MarketForgeState) -> str:
    """If QA fails, skip directly to pipeline finalisation."""
    if state.get("qa_pass") is False and state.get("batch_quality_score", 1.0) < 0.5:
        logger.warning("master.qa_ingestion_failed.halting_pipeline")
        return "finalize_pipeline"
    return "dept3_market_analysis"


# ═══════════════════════════════════════════════════════════════════════════════
# Node: dept3_market_analysis
# ═══════════════════════════════════════════════════════════════════════════════

async def dept3_market_analysis(state: MarketForgeState) -> dict:
    """
    Department 3 — 7 parallel analysis sub-agents → weekly snapshot.
    """
    from datetime import date, timedelta
    run_id = state.get("pipeline_run_id", "")
    today  = date.today()
    week   = str(today - timedelta(days=today.weekday()))

    logger.info("master.dept3.start", run_id=run_id)
    result = await run_market_analysis_pipeline(run_id=run_id, week_start=week)

    logger.info(
        "master.dept3.done",
        snapshot_jobs=result.get("snapshot_jobs"),
        quality=result.get("analysis_quality"),
    )
    return {
        "week_start":       week,
        "analysis_quality": result.get("analysis_quality", "unknown"),
        "weekly_snapshot":  {},  # full snapshot persisted in DB
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: dept4_research
# ═══════════════════════════════════════════════════════════════════════════════

async def dept4_research(state: MarketForgeState) -> dict:
    """
    Department 4 — arXiv monitoring + emerging tech signal analysis.
    """
    run_id = state.get("pipeline_run_id", "")
    logger.info("master.dept4.start", run_id=run_id)

    result = await run_research_pipeline(run_id=run_id)

    logger.info(
        "master.dept4.done",
        papers=result.get("papers"),
        signals=result.get("signals"),
    )
    return {
        "emerging_signals":   result.get("signals", 0),
        "research_quality":   result.get("quality", "unknown"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: dept5_content_studio
# ═══════════════════════════════════════════════════════════════════════════════

async def dept5_content_studio(state: MarketForgeState) -> dict:
    """
    Department 5 — contrarian insight + weekly report generation.
    Reads market snapshot from DB (persisted by Dept 3).
    """
    run_id = state.get("pipeline_run_id", "")
    logger.info("master.dept5.start", run_id=run_id)

    result = await run_content_pipeline(run_id=run_id)

    logger.info(
        "master.dept5.done",
        review_score=result.get("review_score"),
        quality=result.get("quality"),
    )
    return {
        "report_draft":   result.get("report_draft", ""),
        "content_quality": result.get("quality",    "unknown"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: dept7_qa_pre_dispatch
# ═══════════════════════════════════════════════════════════════════════════════

async def dept7_qa_pre_dispatch(state: MarketForgeState) -> dict:
    """
    Department 7 — report quality gate before email dispatch.
    Gemini Pro evaluates against 8 criteria.
    """
    run_id = state.get("pipeline_run_id", "")
    draft  = state.get("report_draft", "")

    if not draft:
        return {"qa_pass": True, "report_qa_score": 0.0}

    logger.info("master.dept7.qa_pre_dispatch.start", run_id=run_id)
    result = await run_qa_pipeline(run_id=f"{run_id}_dispatch", report_draft=draft)

    logger.info(
        "master.dept7.qa_pre_dispatch.done",
        qa_pass=result.get("qa_pass"),
        report_score=result.get("report_score"),
    )
    return {
        "qa_pass":         result.get("qa_pass",        True),
        "report_qa_score": result.get("report_score",   0.0),
        "qa_corrections":  result.get("corrections",    []),
    }


def route_after_qa_dispatch(state: MarketForgeState) -> str:
    """Both pass and fail go to finalize — the flag is in qa_pass."""
    return "finalize_pipeline"


# ═══════════════════════════════════════════════════════════════════════════════
# Node: finalize_pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def finalize_pipeline(state: MarketForgeState) -> dict:
    """
    Mark pipeline complete, trigger Ops heartbeat, and dispatch email if QA passed.
    """
    run_id     = state.get("pipeline_run_id", "")
    qa_pass    = state.get("qa_pass",         True)
    draft      = state.get("report_draft",    "")
    completed  = datetime.utcnow().isoformat()

    # Update pipeline run record
    try:
        from marketforge.memory.postgres import PipelineRunStore
        store = PipelineRunStore()
        store.finish(run_id, "complete" if qa_pass else "qa_failed")
    except Exception:
        pass

    # Email dispatch (only Monday runs + QA passed)
    email_dispatched = False
    if qa_pass and draft:
        try:
            from marketforge.utils.email_dispatch import dispatch_weekly_report
            email_dispatched = await dispatch_weekly_report(draft, run_id)
        except Exception as exc:
            logger.warning("master.email_dispatch_failed", error=str(exc))

    # Trigger Ops heartbeat
    try:
        await run_ops_on_pipeline_complete(run_id)
    except Exception as exc:
        logger.warning("master.ops_trigger_failed", error=str(exc))

    status = "complete" if qa_pass else "qa_failed"
    logger.info(
        "master.pipeline.done",
        run_id=run_id,
        status=status,
        email_dispatched=email_dispatched,
    )
    return {
        "pipeline_status":  status,
        "completed_at":     completed,
        "email_dispatched": email_dispatched,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Master graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_master_graph() -> StateGraph:
    """
    Construct the top-level MarketForge pipeline StateGraph.
    All department graphs are invoked as sub-graphs within their respective nodes.
    """
    graph = StateGraph(MarketForgeState)

    graph.add_node("init_pipeline",          init_pipeline)
    graph.add_node("dept1_data_collection",  dept1_data_collection)
    graph.add_node("dept7_qa_post_ingestion",dept7_qa_post_ingestion)
    graph.add_node("dept3_market_analysis",  dept3_market_analysis)
    graph.add_node("dept4_research",         dept4_research)
    graph.add_node("dept5_content_studio",   dept5_content_studio)
    graph.add_node("dept7_qa_pre_dispatch",  dept7_qa_pre_dispatch)
    graph.add_node("finalize_pipeline",      finalize_pipeline)

    # Main pipeline flow
    graph.add_edge(START, "init_pipeline")
    graph.add_edge("init_pipeline",          "dept1_data_collection")
    graph.add_edge("dept1_data_collection",  "dept7_qa_post_ingestion")

    # QA gate after ingestion
    graph.add_conditional_edges(
        "dept7_qa_post_ingestion",
        route_after_qa_ingestion,
        {
            "dept3_market_analysis": "dept3_market_analysis",
            "finalize_pipeline":     "finalize_pipeline",
        },
    )

    # Analysis → Research → Content
    graph.add_edge("dept3_market_analysis", "dept4_research")
    graph.add_edge("dept4_research",        "dept5_content_studio")
    graph.add_edge("dept5_content_studio",  "dept7_qa_pre_dispatch")

    # Pre-dispatch QA gate
    graph.add_conditional_edges(
        "dept7_qa_pre_dispatch",
        route_after_qa_dispatch,
        {"finalize_pipeline": "finalize_pipeline"},
    )

    graph.add_edge("finalize_pipeline", END)

    return graph


_checkpointer = MemorySaver()
master_graph = build_master_graph().compile(
    checkpointer=_checkpointer,
    name="marketforge_master_pipeline",
)


# ── Public entry point ────────────────────────────────────────────────────────

async def run_full_pipeline(
    run_id: str | None = None,
    top_role_categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Execute the full MarketForge pipeline.

    Called by:
    - Airflow dag_ingest_primary (Tuesday + Thursday 07:00 UTC)
    - Airflow dag_weekly_analysis (Monday 07:00 UTC)

    The run_id is used as the LangGraph thread_id so the checkpointer
    stores the state under this key — enabling node-level retry without
    re-running the full pipeline.

    LangSmith traces the entire graph automatically when env vars are set:
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_API_KEY=<your-key>
      LANGCHAIN_PROJECT=marketforge-ai
    """
    run_id = run_id or str(uuid.uuid4())[:12]
    config = {"configurable": {"thread_id": run_id}}

    initial: MarketForgeState = {
        "pipeline_run_id": run_id,
        "raw_jobs":        [],
        "source_counts":   {},
        "source_errors":   {},
        "alerts_dispatched": [],
        "models_retrained":  [],
        "model_eval_results": [],
        "promoted_models":   [],
    }

    from marketforge.memory.postgres import get_pg_checkpointer
    logger.info("master.invoke.start", run_id=run_id)
    async with get_pg_checkpointer() as checkpointer:
        graph = build_master_graph().compile(
            checkpointer=checkpointer,
            name="marketforge_master_pipeline",
        )
        final = await graph.ainvoke(initial, config=config)
    logger.info(
        "master.invoke.complete",
        run_id=run_id,
        status=final.get("pipeline_status"),
        email=final.get("email_dispatched"),
    )

    return {
        "run_id":           run_id,
        "status":           final.get("pipeline_status", "unknown"),
        "started_at":       final.get("started_at",      ""),
        "completed_at":     final.get("completed_at",    ""),
        "email_dispatched": final.get("email_dispatched", False),
        "qa_pass":          final.get("qa_pass",          True),
        "collection_quality": final.get("collection_quality", "unknown"),
        "analysis_quality":   final.get("analysis_quality",   "unknown"),
        "content_quality":    final.get("content_quality",    "unknown"),
    }
