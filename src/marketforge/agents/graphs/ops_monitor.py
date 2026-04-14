"""
MarketForge AI — Department 9: Ops & Observability LangGraph.

Runs on a 30-minute heartbeat AND on pipeline completion events.

Graph topology:

  START
    ├─► cost_tracker_node      (LLM spend per run/dept/agent)
    ├─► pipeline_health_node   (Airflow DAG state + LangSmith failures)
    └─► infra_health_node      (Postgres pool, Redis memory, ChromaDB sizes)
         │
    merge_ops_results          (fan-in — aggregate all health signals)
         │
    dispatch_alerts            (severity-tiered: SMTP / weekly report / log)
         │
        END
"""
from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import OpsState
from marketforge.agents.ops_monitor.lead_agent import (
    CostTrackerAgent,
    PipelineHealthMonitorAgent,
    InfrastructureHealthAgent,
    AlertDispatchAgent,
    OpsLeadAgent,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel ops health nodes
# ═══════════════════════════════════════════════════════════════════════════════

async def cost_tracker_node(state: OpsState) -> dict:
    """
    CostTrackerAgent — aggregate LLM spend this week.
    Triggers circuit-breaker if spend > 2× rolling average.
    """
    agent  = CostTrackerAgent()
    result = await agent.run({"run_id": state.get("run_id", ""), "trigger": state.get("trigger", "heartbeat")})
    return {"cost_summary": result}


async def pipeline_health_node(state: OpsState) -> dict:
    """
    PipelineHealthMonitorAgent — watches Airflow DAG execution state.
    Detects stalled tasks and cross-references with LangSmith trace failures.
    """
    agent  = PipelineHealthMonitorAgent()
    result = await agent.run({"run_id": state.get("run_id", "")})
    return {"pipeline_health": result}


async def infra_health_node(state: OpsState) -> dict:
    """
    InfrastructureHealthAgent — PostgreSQL pool, Redis memory, ChromaDB sizes.
    Projects capacity headroom 7 days forward.
    """
    agent  = InfrastructureHealthAgent()
    result = await agent.run({"run_id": state.get("run_id", "")})
    return {"infra_health": result}


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-in node — merge_ops_results
# ═══════════════════════════════════════════════════════════════════════════════

async def merge_ops_results(state: OpsState) -> dict:
    """
    Aggregates all parallel ops sub-agent outputs.
    Determines overall ops_quality signal used by AlertDispatchAgent.
    """
    cost     = state.get("cost_summary",    {})
    pipeline = state.get("pipeline_health", {})
    infra    = state.get("infra_health",    {})

    issues: list[str] = []

    # Cost circuit-breaker
    if cost.get("circuit_breaker_triggered"):
        issues.append(f"COST_OVERRUN: ${cost.get('this_week_usd', 0):.3f} > 2× average")

    # Pipeline failures
    failed_dags = pipeline.get("failed_dags", [])
    if failed_dags:
        issues.append(f"PIPELINE_FAILURE: {', '.join(failed_dags)}")

    # Infra pressure
    if infra.get("memory_pressure"):
        issues.append(f"INFRA_MEMORY: Redis at {infra.get('redis_used_pct', 0):.0f}%")
    if infra.get("db_pool_pressure"):
        issues.append(f"INFRA_DB_POOL: {infra.get('pool_utilisation_pct', 0):.0f}% utilised")

    quality = "poor" if any("FAILURE" in i or "OVERRUN" in i for i in issues) \
              else ("warning" if issues else "good")

    logger.info(
        "ops.merge_done",
        quality=quality,
        issues=issues,
        cost_usd=cost.get("this_week_usd", 0),
    )
    return {"ops_quality": quality}


# ═══════════════════════════════════════════════════════════════════════════════
# Alert dispatch node
# ═══════════════════════════════════════════════════════════════════════════════

async def dispatch_alerts(state: OpsState) -> dict:
    """
    AlertDispatchAgent — severity-tiered alerting with 4-hour deduplication.
    Sev 1 (pipeline fail, security breach) → immediate SMTP email
    Sev 2 (model drift, cost overrun warning) → Monday ops report
    Sev 3 (minor anomalies) → logged only
    """
    agent  = AlertDispatchAgent()
    result = await agent.run({
        "run_id":         state.get("run_id", ""),
        "ops_quality":    state.get("ops_quality", "good"),
        "cost_summary":   state.get("cost_summary",    {}),
        "pipeline_health":state.get("pipeline_health", {}),
        "infra_health":   state.get("infra_health",    {}),
    })

    alerts = result.get("dispatched_alerts", [])
    if alerts:
        logger.info("ops.alerts_dispatched", count=len(alerts), severities=[a.get("severity") for a in alerts])

    return {"alerts_dispatched": alerts}


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_ops_graph() -> StateGraph:
    graph = StateGraph(OpsState)

    # Parallel health nodes
    graph.add_node("cost_tracker",    cost_tracker_node)
    graph.add_node("pipeline_health", pipeline_health_node)
    graph.add_node("infra_health",    infra_health_node)

    # Fan-in + dispatch
    graph.add_node("merge_ops_results", merge_ops_results)
    graph.add_node("dispatch_alerts",   dispatch_alerts)

    # Parallel from START
    graph.add_edge(START, "cost_tracker")
    graph.add_edge(START, "pipeline_health")
    graph.add_edge(START, "infra_health")

    # Fan-in
    graph.add_edge("cost_tracker",    "merge_ops_results")
    graph.add_edge("pipeline_health", "merge_ops_results")
    graph.add_edge("infra_health",    "merge_ops_results")

    graph.add_edge("merge_ops_results", "dispatch_alerts")
    graph.add_edge("dispatch_alerts",   END)

    return graph


_checkpointer = MemorySaver()
ops_graph = build_ops_graph().compile(
    checkpointer=_checkpointer,
    name="ops_observability",
)


# ── Entry points ──────────────────────────────────────────────────────────────

async def run_ops_heartbeat(run_id: str | None = None) -> dict[str, Any]:
    """30-minute scheduled heartbeat trigger."""
    import uuid
    run_id = run_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": run_id}}

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_ops_graph().compile(checkpointer=checkpointer, name="ops_observability")
        final = await graph.ainvoke(
            {"run_id": run_id, "trigger": "heartbeat", "alerts_dispatched": []},
            config=config,
        )
    return {
        "run_id":   run_id,
        "quality":  final.get("ops_quality", "unknown"),
        "alerts":   len(final.get("alerts_dispatched", [])),
    }


async def run_ops_on_pipeline_complete(run_id: str) -> dict[str, Any]:
    """Triggered at the end of each ingestion pipeline run."""
    config = {"configurable": {"thread_id": f"ops_{run_id}"}}

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_ops_graph().compile(checkpointer=checkpointer, name="ops_observability")
        final = await graph.ainvoke(
            {"run_id": run_id, "trigger": "pipeline_complete", "alerts_dispatched": []},
            config=config,
        )
    return {
        "run_id":  run_id,
        "quality": final.get("ops_quality", "unknown"),
        "alerts":  len(final.get("alerts_dispatched", [])),
    }
