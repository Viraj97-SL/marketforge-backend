"""
MarketForge AI — Department 2: ML Engineering LangGraph.

Graph topology:

  START
    └─ compute_features      (FeatureEngineeringAgent — PSI drift detection)
         └─ check_drift       (routes to full retrain or light evaluation)
              ├─(drift)─► retrain_models       (parallel: skill_extraction, prescreen, salary, velocity)
              │                └─ register_models    (ModelRegistryAgent — gate promotion)
              └─(no drift)─► evaluate_models   (quick evaluation only)
                                  └─ END
"""
from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from marketforge.agents.graphs.states import MLEngineeringState
from marketforge.agents.ml_engineering.lead_agent import (
    FeatureEngineeringAgent,
    MLEngineerLeadAgent,
    ModelRegistryAgent,
)

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — compute_features
# ═══════════════════════════════════════════════════════════════════════════════

async def compute_features(state: MLEngineeringState) -> dict:
    """
    FeatureEngineeringAgent — compute ML feature matrix for new jobs.
    Detects PSI drift across all numeric features.
    """
    agent  = FeatureEngineeringAgent()
    result = await agent.run({"week_start": state.get("week_start", "")})
    return {
        "feature_count":    result.get("feature_count",    0),
        "drift_report":     result.get("drift_report",     {}),
        "drifted_features": result.get("drifted_features", []),
    }


def route_after_drift_check(state: MLEngineeringState) -> str:
    """If any feature has significant drift (PSI > 0.2), trigger full retrain."""
    if state.get("drifted_features"):
        logger.info(
            "ml_engineering.drift_detected",
            features=state["drifted_features"],
        )
        return "retrain_models"
    return "evaluate_models"


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — retrain_models  (triggered only on drift)
# ═══════════════════════════════════════════════════════════════════════════════

async def retrain_models(state: MLEngineeringState) -> dict:
    """
    Orchestrates retraining of all ML models via MLEngineerLeadAgent.
    Each model trains in the sequence mandated by the spec:
    PreScreen → SkillExtraction → Salary → HiringVelocity
    (sequential so each model can use the improved skill extraction).
    """
    lead   = MLEngineerLeadAgent()
    result = await lead.run({
        "action":          "retrain",
        "drifted_features": state.get("drifted_features", []),
        "week_start":      state.get("week_start", ""),
    })
    return {
        "models_retrained":   result.get("models_retrained",   []),
        "model_eval_results": result.get("model_eval_results", []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — evaluate_models  (no drift — quick check only)
# ═══════════════════════════════════════════════════════════════════════════════

async def evaluate_models(state: MLEngineeringState) -> dict:
    """
    Quick evaluation when no significant drift detected.
    Checks production models haven't regressed since last retrain.
    """
    lead   = MLEngineerLeadAgent()
    result = await lead.run({
        "action":     "evaluate",
        "week_start": state.get("week_start", ""),
    })
    return {"model_eval_results": result.get("model_eval_results", [])}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — register_models  (post-retrain gate)
# ═══════════════════════════════════════════════════════════════════════════════

async def register_models(state: MLEngineeringState) -> dict:
    """
    ModelRegistryAgent — multi-metric gate for production promotion.
    A model is promoted only if:
    - Primary metric improves vs current production
    - Secondary metrics don't regress
    - Fairness check passes across company_stage categories
    - Training data hash recorded in MLflow
    """
    agent  = ModelRegistryAgent()
    result = await agent.run({
        "models_retrained":   state.get("models_retrained",   []),
        "model_eval_results": state.get("model_eval_results", []),
    })
    return {
        "promoted_models": result.get("promoted_models", []),
        "ml_quality":      result.get("quality", "good"),
    }


async def finalize_ml(state: MLEngineeringState) -> dict:
    """Finalize — set ml_quality if not already set."""
    quality = state.get("ml_quality", "good")
    promoted = state.get("promoted_models", [])
    logger.info("ml_engineering.done", promoted=promoted, quality=quality)
    return {"ml_quality": quality}


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_ml_engineering_graph() -> StateGraph:
    graph = StateGraph(MLEngineeringState)

    graph.add_node("compute_features", compute_features)
    graph.add_node("retrain_models",   retrain_models)
    graph.add_node("evaluate_models",  evaluate_models)
    graph.add_node("register_models",  register_models)
    graph.add_node("finalize_ml",      finalize_ml)

    graph.add_edge(START, "compute_features")

    # Conditional: drift detected → retrain; no drift → evaluate
    graph.add_conditional_edges(
        "compute_features",
        route_after_drift_check,
        {"retrain_models": "retrain_models", "evaluate_models": "evaluate_models"},
    )

    graph.add_edge("retrain_models",  "register_models")
    graph.add_edge("register_models", "finalize_ml")
    graph.add_edge("evaluate_models", "finalize_ml")
    graph.add_edge("finalize_ml",     END)

    return graph


_checkpointer = MemorySaver()
ml_engineering_graph = build_ml_engineering_graph().compile(
    checkpointer=_checkpointer,
    name="ml_engineering",
)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_ml_pipeline(
    run_id: str | None = None,
    week_start: str | None = None,
) -> dict[str, Any]:
    """Called by the Airflow dag_model_retrain DAG."""
    import uuid
    run_id = run_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": run_id}}

    initial: MLEngineeringState = {
        "run_id":     run_id,
        "week_start": week_start or "",
        "models_retrained":   [],
        "model_eval_results": [],
        "promoted_models":    [],
    }

    from marketforge.memory.postgres import get_pg_checkpointer
    async with get_pg_checkpointer() as checkpointer:
        graph = build_ml_engineering_graph().compile(
            checkpointer=checkpointer,
            name="ml_engineering",
        )
        final = await graph.ainvoke(initial, config=config)

    return {
        "run_id":          run_id,
        "features_computed": final.get("feature_count", 0),
        "drifted_features":  final.get("drifted_features", []),
        "models_retrained":  final.get("models_retrained", []),
        "promoted_models":   final.get("promoted_models", []),
        "quality":           final.get("ml_quality", "unknown"),
    }
