"""
MarketForge AI — LangGraph State Schema.

The TypedDict that flows through the DAG:
  DataCollection → NLPExtraction → MarketAnalysis → ContentStudio
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from marketforge.models.job import EnrichedJob, MarketSnapshot, PipelineRun, RawJob


class DataCollectionState(TypedDict, total=False):
    """State fields owned by the Data Collection department."""
    run_id:           str
    run_started_at:   str
    raw_jobs:         list[RawJob]
    deduped_jobs:     list[RawJob]
    source_counts:    dict[str, int]   # connector_name → jobs_returned
    source_errors:    dict[str, str]   # connector_name → error_message
    dedup_report:     dict[str, Any]   # from DeduplicationCoordinatorAgent


class NLPState(TypedDict, total=False):
    """State fields owned by the NLP extraction Airflow task."""
    enriched_jobs:      list[EnrichedJob]
    skill_extraction_stats: dict[str, int]  # gate1_hits / gate2_hits / gate3_hits / total


class AnalysisState(TypedDict, total=False):
    """State fields owned by the Market Analysis department."""
    weekly_snapshot:    MarketSnapshot
    skill_trends:       dict[str, Any]
    salary_stats:       dict[str, Any]
    geo_distribution:   dict[str, Any]
    tech_fingerprints:  list[dict]


class ContentState(TypedDict, total=False):
    """State fields owned by the Content Studio department."""
    report_draft:     str
    report_final:     str
    chart_annotations: list[str]
    email_dispatched: bool


class MarketForgeState(
    DataCollectionState,
    NLPState,
    AnalysisState,
    ContentState,
    total=False,
):
    """
    Master state object that flows through the entire LangGraph DAG.
    Fields are typed strictly — every department reads only what it needs
    and writes only what it owns. No raw dicts.
    """
    pipeline_run: PipelineRun
