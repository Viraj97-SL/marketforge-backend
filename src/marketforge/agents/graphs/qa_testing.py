"""
MarketForge AI — Department 7: QA & Testing LangGraph.

Graph topology:

  START
    ├─► data_integrity_node       (post-ingestion batch checks)
    ├─► connector_health_node     (canary requests to all job sources)
    └─► model_drift_node          (PSI on production model inputs)
         │
    merge_qa_results              (fan-in — aggregate all health signals)
         │
    ─ report_quality_node         (pre-dispatch report evaluation — LLM)
    (if report_draft in state)
         │
    qa_decision                   (pass/fail gate — blocks email dispatch)
         │
        END
"""
from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import QAState
from marketforge.agents.qa_testing.lead_agent import (
    DataIntegrityAgent,
    ReportQualityAgent,
    QALeadAgent,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel QA nodes
# ═══════════════════════════════════════════════════════════════════════════════

async def data_integrity_node(state: QAState) -> dict:
    """
    DataIntegrityAgent — 8 schema + semantic checks on the current ingestion batch.
    Null rates, salary anomalies, duplicate detection, description length.
    """
    agent  = DataIntegrityAgent()
    result = await agent.run({"run_id": state.get("run_id", "")})
    return {
        "data_integrity_result": result,
        "batch_quality_score":   result.get("batch_quality_score", 0.0),
    }


async def connector_health_node(state: QAState) -> dict:
    """
    ConnectorHealthAgent — lightweight canary requests to each job source.
    Checks HTTP status + parse rate on 5 samples per source.
    """
    from marketforge.agents.qa_testing.lead_agent import ConnectorHealthAgent
    agent  = ConnectorHealthAgent()
    result = await agent.run({"run_id": state.get("run_id", "")})
    return {"connector_health_scores": result.get("health_scores", {})}


async def model_drift_node(state: QAState) -> dict:
    """
    ModelDriftDetectionAgent — PSI on production model input distributions.
    Triggers MLEngineerLeadAgent if drift_risk_score > 0.2.
    """
    from marketforge.agents.qa_testing.lead_agent import ModelDriftDetectionAgent
    agent  = ModelDriftDetectionAgent()
    result = await agent.run({"run_id": state.get("run_id", "")})
    return {"drift_risk_scores": result.get("drift_risk_scores", {})}


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-in node — merge_qa_results
# ═══════════════════════════════════════════════════════════════════════════════

async def merge_qa_results(state: QAState) -> dict:
    """
    Aggregates outputs from all parallel QA sub-agents.
    Runs after ALL three parallel nodes complete (LangGraph fan-in).
    """
    batch_score    = state.get("batch_quality_score",    0.0)
    health_scores  = state.get("connector_health_scores", {})
    drift_scores   = state.get("drift_risk_scores",      {})

    # Overall health: avg connector health > 0.7 and batch score > 0.8
    avg_health   = sum(health_scores.values()) / max(len(health_scores), 1) if health_scores else 1.0
    max_drift    = max(drift_scores.values(), default=0.0) if drift_scores else 0.0
    qa_quality   = "good" if (avg_health > 0.7 and batch_score > 0.8 and max_drift < 0.2) else "warning"

    # Flag struggling connectors
    low_health   = [src for src, score in health_scores.items() if score < 0.5]
    if low_health:
        logger.warning("qa.connector_health_low", sources=low_health)

    if max_drift >= 0.2:
        logger.warning("qa.model_drift_detected", max_psi=max_drift)

    logger.info(
        "qa.merge_done",
        batch_score=batch_score,
        avg_connector_health=round(avg_health, 2),
        max_drift=round(max_drift, 3),
        quality=qa_quality,
    )
    return {"qa_quality": qa_quality}


# ═══════════════════════════════════════════════════════════════════════════════
# Report quality node  (only runs if report_draft in state)
# ═══════════════════════════════════════════════════════════════════════════════

async def report_quality_node(state: QAState) -> dict:
    """
    ReportQualityAgent — evaluates the ContentStudio draft against 8 criteria
    using Gemini Pro as an evaluator.  Runs only when a report draft is present.
    """
    draft = state.get("report_draft", "")
    if not draft:
        return {"report_qa_score": 0.0, "qa_pass": True, "qa_corrections": []}

    agent  = ReportQualityAgent()
    result = await agent.run({"report_draft": draft})

    score       = result.get("overall_score",  0.0)
    passed      = result.get("pass",           score >= 7.5)
    corrections = result.get("corrections",    [])

    logger.info(
        "qa.report_quality.done",
        score=score,
        pass_=passed,
        corrections=len(corrections),
    )
    return {
        "report_qa_score": score,
        "qa_pass":         passed,
        "qa_corrections":  corrections,
    }


def route_to_report_qa(state: QAState) -> str:
    return "report_quality" if state.get("report_draft") else "qa_decision"


# ═══════════════════════════════════════════════════════════════════════════════
# Final decision node
# ═══════════════════════════════════════════════════════════════════════════════

async def qa_decision(state: QAState) -> dict:
    """
    Final QA gate.  Aggregates all quality signals into a single pass/fail.
    A failed QA gate blocks email dispatch (handled in ContentStudio).
    """
    overall_quality = state.get("qa_quality", "good")
    report_pass     = state.get("qa_pass",     True)
    corrections     = state.get("qa_corrections", [])

    final_pass = (overall_quality != "poor") and report_pass

    logger.info(
        "qa.decision",
        final_pass=final_pass,
        overall_quality=overall_quality,
        report_pass=report_pass,
        corrections=len(corrections),
    )
    return {
        "qa_pass":    final_pass,
        "qa_quality": "good" if final_pass else "warning",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_qa_graph() -> StateGraph:
    graph = StateGraph(QAState)

    # Parallel health-check nodes
    graph.add_node("data_integrity",    data_integrity_node)
    graph.add_node("connector_health",  connector_health_node)
    graph.add_node("model_drift",       model_drift_node)

    # Fan-in + optional report QA
    graph.add_node("merge_qa_results",  merge_qa_results)
    graph.add_node("report_quality",    report_quality_node)
    graph.add_node("qa_decision",       qa_decision)

    # Parallel from START
    graph.add_edge(START, "data_integrity")
    graph.add_edge(START, "connector_health")
    graph.add_edge(START, "model_drift")

    # Fan-in
    graph.add_edge("data_integrity",   "merge_qa_results")
    graph.add_edge("connector_health", "merge_qa_results")
    graph.add_edge("model_drift",      "merge_qa_results")

    # Conditional: report present → evaluate it; else skip
    graph.add_conditional_edges(
        "merge_qa_results",
        route_to_report_qa,
        {"report_quality": "report_quality", "qa_decision": "qa_decision"},
    )
    graph.add_edge("report_quality", "qa_decision")
    graph.add_edge("qa_decision",    END)

    return graph


_checkpointer = MemorySaver()
qa_graph = build_qa_graph().compile(
    checkpointer=_checkpointer,
    name="qa_testing",
)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_qa_pipeline(
    run_id: str | None = None,
    report_draft: str | None = None,
) -> dict[str, Any]:
    """Called after ingestion and before email dispatch."""
    import uuid
    run_id = run_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": run_id}}

    initial: QAState = {
        "run_id":       run_id,
        "report_draft": report_draft or "",
    }

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_qa_graph().compile(
            checkpointer=checkpointer,
            name="qa_testing",
        )
        final = await graph.ainvoke(initial, config=config)

    return {
        "run_id":         run_id,
        "qa_pass":        final.get("qa_pass",        True),
        "batch_score":    final.get("batch_quality_score", 0.0),
        "report_score":   final.get("report_qa_score", 0.0),
        "corrections":    final.get("qa_corrections",  []),
        "quality":        final.get("qa_quality",      "unknown"),
    }
