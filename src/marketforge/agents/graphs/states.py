"""
MarketForge AI — LangGraph State Definitions (v2, full LangGraph refactor).

Every department owns a TypedDict slice.  Annotated fields use reducer
functions so parallel fan-out nodes can safely accumulate results without
clobbering each other.

The master pipeline passes `MarketForgeState` (all slices combined) through
the top-level StateGraph.  Per-request graphs (UserInsights) use their own
isolated state.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


# ── Reducer helpers ──────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts — later writers win on key conflicts."""
    return {**a, **b}


def _merge_lists(a: list, b: list) -> list:
    """Concatenate two lists (same as operator.add)."""
    return a + b


# ═══════════════════════════════════════════════════════════════════════════════
# Department 1 — Data Collection
# ═══════════════════════════════════════════════════════════════════════════════

class DataCollectionState(TypedDict, total=False):
    """
    Flows through the DataCollection StateGraph.

    raw_jobs / source_counts / source_errors use reducers because multiple
    scraper nodes write to them in parallel via the Send fan-out.
    """
    run_id:           str
    active_sources:   list[str]          # decided by plan_collection node
    scraper_context:  dict[str, Any]     # forwarded to every scraper node

    # Accumulated across parallel scraper fan-out
    raw_jobs:        Annotated[list[dict], operator.add]
    source_counts:   Annotated[dict[str, int], _merge_dicts]
    source_errors:   Annotated[dict[str, str], _merge_dicts]

    # Set by deduplication node
    deduped_jobs:    list[dict]
    dedup_report:    dict[str, Any]

    # Set by reflect node
    collection_quality: str
    adaptive_params:    dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 2 — ML Engineering
# ═══════════════════════════════════════════════════════════════════════════════

class MLEngineeringState(TypedDict, total=False):
    run_id:               str
    week_start:           str
    feature_count:        int
    drift_report:         dict[str, Any]
    drifted_features:     list[str]
    models_retrained:     Annotated[list[str], operator.add]
    model_eval_results:   Annotated[list[dict], operator.add]
    promoted_models:      Annotated[list[str], operator.add]
    ml_quality:           str
    adaptive_params:      dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 3 — Market Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class MarketAnalysisState(TypedDict, total=False):
    run_id:         str
    week_start:     str

    # Each analysis sub-agent writes one of these keys
    skill_trends:       dict[str, Any]      # SkillDemandAnalystAgent
    salary_stats:       dict[str, Any]      # SalaryIntelligenceAgent
    sponsorship_data:   dict[str, Any]      # SponsorshipTrackerAgent
    velocity_data:      dict[str, Any]      # HiringVelocityAgent
    cooccurrence_pairs: list[dict]          # SkillCoOccurrenceAgent
    geo_distribution:   dict[str, Any]      # GeographicDistributionAgent
    tech_archetypes:    list[dict]          # TechStackFingerprintAgent

    # Assembled by the compile_snapshot node
    weekly_snapshot:    dict[str, Any]
    analysis_quality:   str
    adaptive_params:    dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 4 — Research Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchState(TypedDict, total=False):
    run_id:               str
    top_skills:           dict[str, int]    # passed in from MarketAnalysis

    research_papers:      list[dict]        # arXivMonitorAgent
    summary_cards:        list[dict]        # arXivMonitorAgent (LLM cards)
    emerging_signals:     list[dict]        # EmergingTechSignalAgent
    confirmed_adoptions:  list[dict]        # EmergingTechSignalAgent
    mean_adoption_lag_days: float

    research_quality:     str
    adaptive_params:      dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 5 — Content Studio
# ═══════════════════════════════════════════════════════════════════════════════

class ContentStudioState(TypedDict, total=False):
    run_id:              str
    snapshot:            dict[str, Any]   # from MarketAnalysis
    emerging_signals:    list[dict]       # from Research

    contrarian_insight:  str              # ContrarianInsightAgent
    report_draft:        str              # WeeklyReportWriterAgent
    self_review_score:   float
    report_final:        str              # post-QA revision
    email_dispatched:    bool

    content_quality:     str
    adaptive_params:     dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 6 — User Career Insights  (per-request, isolated)
# ═══════════════════════════════════════════════════════════════════════════════

class UserInsightsState(TypedDict, total=False):
    """Stateless per HTTP request — never persisted."""
    sanitised_profile:   dict[str, Any]
    visa_needed:         bool

    # Set by plan_insights node (deterministic)
    user_skills:         list[str]
    target_role:         str
    exp_level:           str
    skill_gaps:          list[dict]
    match_score:         float
    market_snapshot:     dict[str, Any]
    sector_fit:          list[dict]

    # Set by synthesise_narrative node (LLM)
    career_narrative:    str
    action_plan:         str

    # Security gate flags
    security_passed:     bool
    rejection_reason:    str
    insights_quality:    str


# ═══════════════════════════════════════════════════════════════════════════════
# Department 7 — QA & Testing
# ═══════════════════════════════════════════════════════════════════════════════

class QAState(TypedDict, total=False):
    run_id:               str
    report_draft:         str             # from ContentStudio

    # Each QA sub-agent writes one result
    data_integrity_result:   dict[str, Any]
    report_quality_result:   dict[str, Any]
    llm_output_samples:      list[dict]
    connector_health_scores: dict[str, float]
    drift_risk_scores:       dict[str, float]

    # Gate decision
    qa_pass:              bool
    batch_quality_score:  float
    report_qa_score:      float
    qa_corrections:       list[str]
    qa_quality:           str
    adaptive_params:      dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 8 — Security & Guardrails  (synchronous middleware)
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityState(TypedDict, total=False):
    """
    Flows through the Security StateGraph.
    Security runs synchronously — always linear, never parallel.
    """
    raw_input:        dict[str, Any]
    operation_type:   str      # "career_advice" | "content_dispatch" | "ingestion"

    # Set by sanitise_input node
    sanitised_input:  dict[str, Any]
    input_rejected:   bool
    rejection_code:   str

    # Set by detect_injection node
    injection_score:  float
    injection_flagged: bool

    # Set by scrub_pii node
    pii_types_found:  list[str]
    scrubbed_output:  dict[str, Any]

    # Set by validate_output node
    output_validated: bool
    unverifiable_claims: list[str]

    # Set by threat_intel node
    threat_level:     str   # "low" | "medium" | "high"
    threat_logged:    bool

    security_passed:  bool
    adaptive_params:  dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Department 9 — Ops & Observability
# ═══════════════════════════════════════════════════════════════════════════════

class OpsState(TypedDict, total=False):
    run_id:               str
    trigger:              str    # "heartbeat" | "pipeline_complete" | "alert_event"

    # Written by parallel health-check nodes
    cost_summary:         dict[str, Any]
    pipeline_health:      dict[str, Any]
    infra_health:         dict[str, Any]
    benchmark_results:    dict[str, Any]

    # Assembled by dispatch_alerts node
    alerts_dispatched:    Annotated[list[dict], operator.add]
    ops_report:           str

    ops_quality:          str
    adaptive_params:      dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Master Pipeline State  (combines all department slices)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketForgeState(
    DataCollectionState,
    MLEngineeringState,
    MarketAnalysisState,
    ResearchState,
    ContentStudioState,
    QAState,
    OpsState,
    total=False,
):
    """
    The master state object that flows through the top-level pipeline graph.

    Each department Lead Agent node reads only the keys it needs and writes
    only the keys it owns.  The LangGraph engine handles state persistence
    via the configured checkpointer (PostgreSQL in production, MemorySaver
    in development/testing).
    """
    pipeline_run_id:  str
    pipeline_status:  str    # "running" | "complete" | "failed"
    started_at:       str
    completed_at:     str
