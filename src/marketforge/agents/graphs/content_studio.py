"""
MarketForge AI — Department 5: Content Studio LangGraph.

Graph topology:

  START
    └─ load_content_context     (pulls weekly snapshot + emerging signals)
         └─ generate_contrarian  (ContrarianInsightAgent — Flash LLM)
              └─ write_report     (WeeklyReportWriterAgent — Pro LLM, 2-pass)
                   └─ self_review  (quality gate: score < 7.5 → rewrite flag)
                        └─ END

Linear graph — each stage depends on the previous stage's output.
LLM calls: Gemini Flash for contrarian, Gemini Pro for report assembly.
"""
from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import ContentStudioState
from marketforge.agents.content_studio.lead_agent import WeeklyReportWriterAgent

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — load_content_context
# ═══════════════════════════════════════════════════════════════════════════════

async def load_content_context(state: ContentStudioState) -> dict:
    """
    Load the latest weekly snapshot and emerging research signals from DB.
    These are the ONLY inputs to the LLM — no free-form text injection.
    """
    import json
    snapshot: dict[str, Any] = state.get("snapshot", {})

    if not snapshot:
        try:
            from marketforge.memory.postgres import get_sync_engine
            from sqlalchemy import text

            engine  = get_sync_engine()
            snaps_t = "weekly_snapshots" if engine.dialect.name == "sqlite" else "market.weekly_snapshots"

            with engine.connect() as conn:
                row = conn.execute(text(f"""
                    SELECT top_skills, salary_p25, salary_p50, salary_p75,
                           salary_sample_size, sponsorship_rate, remote_rate,
                           job_count, week_start
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
                    "remote_rate":        row[6],
                    "job_count":          row[7],
                    "week_start":         str(row[8]) if row[8] else "",
                }
        except Exception as exc:
            logger.warning("content.context_load_failed", error=str(exc))

    # Emerging signals from research department (already in state from master graph)
    emerging = state.get("emerging_signals", [])

    logger.info("content.context_loaded", jobs=snapshot.get("job_count", 0), signals=len(emerging))
    return {"snapshot": snapshot, "emerging_signals": emerging}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — generate_contrarian
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_contrarian(state: ContentStudioState) -> dict:
    """
    ContrarianInsightAgent — uses Gemini Flash to find the one surprising
    pattern that a straightforward summary would miss.
    """
    import json
    from marketforge.config.settings import settings

    snap = state.get("snapshot", {})
    if not snap:
        return {"contrarian_insight": "Insufficient data for contrarian analysis this week."}

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        _SYSTEM = """You are a data analyst who finds counterintuitive patterns.
Given market data, identify ONE surprising observation a straightforward summary would miss.
Rules: statistically defensible, genuinely counterintuitive, actionable.
Output: one paragraph, max 80 words."""

        prompt = f"""Market data week of {snap.get('week_start')}:
Top skills: {json.dumps(list(snap.get('top_skills', {}).items())[:10])}
Salary P50: £{snap.get('salary_p50', 0):,}
Sponsorship rate: {round((snap.get('sponsorship_rate') or 0)*100, 1)}%
Remote rate: {round((snap.get('remote_rate') or 0)*100, 1)}%
Find the ONE most counterintuitive observation."""

        flash = ChatGoogleGenerativeAI(
            model=settings.llm.fast_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.3,
        )
        msg = await flash.ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
        insight = msg.content.strip()
        logger.info("content.contrarian.done", words=len(insight.split()))
        return {"contrarian_insight": insight}

    except Exception as exc:
        logger.warning("content.contrarian.failed", error=str(exc))
        return {"contrarian_insight": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — write_report
# ═══════════════════════════════════════════════════════════════════════════════

async def write_report(state: ContentStudioState) -> dict:
    """
    WeeklyReportWriterAgent — full 2-pass LLM report generation.
    Pass 1: identify significant week-over-week changes.
    Pass 2: Gemini Pro assembles the final structured report.
    """
    agent  = WeeklyReportWriterAgent()
    result = await agent.run({
        "snapshot":          state.get("snapshot", {}),
        "contrarian_insight": state.get("contrarian_insight", ""),
        "emerging_signals":  state.get("emerging_signals", []),
    })
    return {
        "report_draft":      result.get("report_draft",      ""),
        "self_review_score": result.get("self_review_score", 0.0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — self_review
# ═══════════════════════════════════════════════════════════════════════════════

async def self_review(state: ContentStudioState) -> dict:
    """
    Quality gate: evaluate the draft against 7 criteria.
    self_review_score < 7.5 → content_quality = "warning" (triggers QA rewrite).
    """
    draft = state.get("report_draft", "")
    score = state.get("self_review_score", 0.0)

    # Lightweight heuristics if LLM score wasn't set
    if score == 0.0:
        words      = len(draft.split())
        has_nums   = any(c.isdigit() for c in draft)
        score = 8.0
        if words < 200: score -= 3
        elif words > 700: score -= 1
        if not has_nums: score -= 2

    quality = "good" if score >= 7.5 else "warning"

    logger.info(
        "content.self_review.done",
        score=score,
        words=len(draft.split()),
        quality=quality,
    )
    return {
        "self_review_score": score,
        "content_quality":   quality,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_content_studio_graph() -> StateGraph:
    graph = StateGraph(ContentStudioState)

    graph.add_node("load_content_context", load_content_context)
    graph.add_node("generate_contrarian",  generate_contrarian)
    graph.add_node("write_report",         write_report)
    graph.add_node("self_review",          self_review)

    graph.add_edge(START,                  "load_content_context")
    graph.add_edge("load_content_context", "generate_contrarian")
    graph.add_edge("generate_contrarian",  "write_report")
    graph.add_edge("write_report",         "self_review")
    graph.add_edge("self_review",          END)

    return graph


_checkpointer = MemorySaver()
content_studio_graph = build_content_studio_graph().compile(
    checkpointer=_checkpointer,
    name="content_studio",
)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_content_pipeline(
    run_id: str | None = None,
    snapshot: dict | None = None,
    emerging_signals: list | None = None,
) -> dict[str, Any]:
    """Called by the Airflow generate_content_draft task."""
    import uuid
    run_id = run_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": run_id}}

    initial: ContentStudioState = {
        "run_id":          run_id,
        "snapshot":        snapshot or {},
        "emerging_signals": emerging_signals or [],
    }

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_content_studio_graph().compile(
            checkpointer=checkpointer,
            name="content_studio",
        )
        final = await graph.ainvoke(initial, config=config)

    return {
        "run_id":         run_id,
        "report_draft":   final.get("report_draft", ""),
        "review_score":   final.get("self_review_score", 0.0),
        "quality":        final.get("content_quality", "unknown"),
    }
