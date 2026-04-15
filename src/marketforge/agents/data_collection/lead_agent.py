"""
MarketForge AI — Department 1: DataCollectionLeadAgent

The Lead Agent orchestrates all data collection sub-agents.
It plans the run strategy based on prior yields, dispatches sub-agents
in a parallel fan-out, merges results, runs cross-source deduplication,
validates corpus health, and writes to market.jobs.

On every run it compares yield patterns against a 4-week rolling baseline
and flags degradation to the Ops department.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
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
from marketforge.memory.postgres import AgentStateStore, JobStore, PipelineRunStore, init_database
from marketforge.models.job import RawJob
from marketforge.utils.cost_tracker import CostTracker

logger = structlog.get_logger(__name__)


class DataCollectionLeadAgent(DeepAgent):
    """
    Department 1 Lead Agent — Data Collection.

    Responsibilities:
    - Plans run strategy: reads 4-week yield history to detect degrading sources
    - Dispatches 9 sub-agents in a fan-out (parallel execution)
    - Merges raw results from all sub-agents
    - Delegates cross-source deduplication to DeduplicationCoordinatorAgent
    - Validates corpus health (minimum job count, source diversity)
    - Writes all new jobs to market.jobs via JobStore
    - Reports telemetry back to the pipeline orchestrator

    The lead agent does NOT do any scraping itself — that is sub-agent territory.
    """

    agent_id   = "data_collection_lead_v1"
    department = "data_collection"

    def __init__(self) -> None:
        # Instantiate all sub-agents — all 9 registered from day 1
        self._sub_agents: dict[str, DeepAgent] = {
            # Phase 1: API-based sources (active immediately)
            "adzuna":           AdzunaDeepScoutAgent(),
            "reed":             ReedDeepScoutAgent(),
            # Phase 1: Additional sources
            "wellfound":        WellfoundDeepScoutAgent(),
            "ats_direct":       ATSDirectDeepAgent(),
            "career_pages":     CareerPagesDeepCrawlerAgent(),
            "funding_news":     FundingNewsDeepDiscoveryAgent(),
            "recruiter_boards": RecruiterBoardsDeepAgent(),
            "specialist_boards":SpecialistBoardsDeepAgent(),
        }
        self._dedup_agent = DeduplicationCoordinatorAgent()
        self._job_store   = JobStore()

    # ── PLAN ─────────────────────────────────────────────────────────────────

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse prior-run yield history per source.
        Sources with 3+ consecutive zero-yield runs are skipped.
        Sources with yield > 2× the 4-week average get a higher concurrency slot.
        """
        adaptive       = state.get("adaptive_params", {})
        source_yields  = adaptive.get("source_yields", {})   # {source: [yield_w1, w2, w3, w4]}
        disabled       = set(adaptive.get("disabled_sources", []))

        # Determine which sub-agents to dispatch
        active_agents = [
            name for name in self._sub_agents.keys()
            if name not in disabled
        ]

        # Log any sources that have degraded
        degraded = []
        for src, yields in source_yields.items():
            if len(yields) >= 3 and all(y == 0 for y in yields[-3:]):
                degraded.append(src)
                logger.warning("lead.source_degraded", source=src, yields=yields[-3:])

        top_role_categories = context.get("top_role_categories", [])

        logger.info(
            f"{self.agent_id}.plan.done",
            active_sources=active_agents,
            degraded=degraded,
        )
        return {
            "active_agents":       active_agents,
            "top_role_categories": top_role_categories,
            "adaptive":            adaptive,
        }

    # ── EXECUTE ───────────────────────────────────────────────────────────────

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """
        Fan-out: run all active sub-agents in parallel.
        Each sub-agent runs its own Plan→Execute→Reflect lifecycle.
        Results are merged, then deduplication coordinator runs.
        """
        active    = plan["active_agents"]
        top_cats  = plan["top_role_categories"]
        run_id    = state.get("run_id", str(uuid.uuid4())[:8])

        # Build per-agent context (sub-agents can read top role categories
        # to adjust their own query matrices)
        sub_context = {
            "top_role_categories_by_demand": top_cats,
            "run_id":                        run_id,
            "watchlist":                     state.get("watchlist", []),
        }

        # Dispatch all sub-agents concurrently
        sem = asyncio.Semaphore(9)  # all 9 can run in parallel

        async def run_sub(name: str) -> tuple[str, dict]:
            async with sem:
                agent = self._sub_agents[name]
                try:
                    result = await agent.run(sub_context)
                    return name, result
                except Exception as exc:
                    logger.error(f"lead.sub_agent_failed", agent=name, error=str(exc))
                    return name, {"jobs": [], "error": str(exc)}

        t0 = time.monotonic()
        sub_results = await asyncio.gather(*[run_sub(n) for n in active])
        elapsed = round(time.monotonic() - t0, 2)

        # Merge all job lists
        all_raw: list[RawJob] = []
        source_counts: dict[str, int] = {}
        source_errors: dict[str, str] = {}

        for name, result in sub_results:
            jobs = result.get("jobs", [])
            source_counts[name] = len(jobs)
            if "error" in result:
                source_errors[name] = result["error"]
            all_raw.extend(jobs)

        logger.info(
            f"{self.agent_id}.fan_out.done",
            total_raw=len(all_raw),
            sources=source_counts,
            elapsed_s=elapsed,
        )

        # ── Deduplication ─────────────────────────────────────────────────────
        dedup_context = {"raw_jobs": all_raw, "run_id": run_id}
        dedup_result  = await self._dedup_agent.run(dedup_context)
        deduped_jobs  = dedup_result.get("deduped_jobs", [])
        dedup_report  = dedup_result.get("dedup_report", {})

        logger.info(
            f"{self.agent_id}.dedup.done",
            raw=len(all_raw),
            deduped=len(deduped_jobs),
            removed=dedup_report.get("removed", 0),
        )

        # ── Refresh scraped_at for ALL jobs seen this run ─────────────────────
        # Re-scraped jobs (same dedup_hash, filtered out above) still exist in
        # market.jobs with a stale scraped_at from their first scrape run.
        # Touching scraped_at ensures market analysis queries (scraped_at >= week_start)
        # count all currently-live jobs, not just first-seen jobs.
        all_raw_job_ids = [j.job_id for j in all_raw]
        try:
            touched = self._job_store.touch_scraped_at(all_raw_job_ids)
            logger.info(f"{self.agent_id}.touch_scraped_at", touched=touched, total_raw=len(all_raw_job_ids))
        except Exception as exc:
            logger.warning(f"{self.agent_id}.touch_scraped_at_failed", error=str(exc))

        # ── Write new jobs to database ────────────────────────────────────────
        write_errors = 0
        for job in deduped_jobs:
            try:
                self._job_store.upsert_job(job, run_id)
            except Exception as exc:
                write_errors += 1
                logger.warning("lead.write_error", job_id=job.job_id, error=str(exc))

        if write_errors:
            logger.warning(f"{self.agent_id}.write_errors", count=write_errors)

        return {
            "raw_jobs":      all_raw,
            "deduped_jobs":  deduped_jobs,
            "source_counts": source_counts,
            "source_errors": source_errors,
            "dedup_report":  dedup_report,
            "write_errors":  write_errors,
            "elapsed_s":     elapsed,
        }

    # ── REFLECT ───────────────────────────────────────────────────────────────

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare this run's yields against the 4-week rolling baseline.
        Detect corpus health issues:
          - Total yield < 50% of 4-week average → "warning"
          - Any single source at 0 yield for 3 runs → flag for disable
          - Source diversity < 2 active sources → "poor"
        """
        adaptive      = plan.get("adaptive", {})
        source_counts = result.get("source_counts", {})
        deduped       = result.get("deduped_jobs", [])

        # Update 4-week rolling yield history
        source_yields: dict[str, list[int]] = adaptive.get("source_yields", {})
        for src, count in source_counts.items():
            history = source_yields.get(src, [])
            history.append(count)
            source_yields[src] = history[-4:]   # keep last 4 runs

        # Sources with 3+ zero-yield runs → candidate for disabling
        disabled = list(adaptive.get("disabled_sources", []))
        for src, history in source_yields.items():
            if len(history) >= 3 and all(y == 0 for y in history[-3:]):
                if src not in disabled:
                    disabled.append(src)
                    logger.warning(f"{self.agent_id}.disabling_source", source=src)

        adaptive["source_yields"]    = source_yields
        adaptive["disabled_sources"] = disabled
        state["adaptive_params"]     = adaptive
        state["last_yield"]          = len(deduped)

        # Quality gate
        total_yield = len(deduped)
        n_active    = len([c for c in source_counts.values() if c > 0])

        quality = "good"
        if n_active < 2:
            quality = "poor"
        elif total_yield < 30:
            quality = "warning"

        return {
            "quality":     quality,
            "yield":       total_yield,
            "n_sources":   n_active,
            "source_counts": source_counts,
            "notes":       f"deduped={total_yield}, sources_active={n_active}",
        }

    # ── OUTPUT ────────────────────────────────────────────────────────────────

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "raw_jobs":      result.get("raw_jobs",      []),
            "deduped_jobs":  result.get("deduped_jobs",  []),
            "source_counts": result.get("source_counts", {}),
            "source_errors": result.get("source_errors", {}),
            "dedup_report":  result.get("dedup_report",  {}),
            "collection_quality": reflection.get("quality", "unknown"),
        }


# ── Module-level entry point for Airflow ────────────────────────────────────

async def run_data_collection(run_id: str, cost_tracker: CostTracker | None = None) -> dict:
    """
    Called by the Airflow scrape_all_sources task.
    Initialises DB, runs the lead agent, returns a summary dict for XCom.
    """
    init_database()
    run_store = PipelineRunStore()
    run_store.start(run_id, "dag_ingest_primary")

    lead = DataCollectionLeadAgent()
    # Prime the agent state with the run_id
    context = {"run_id": run_id}

    try:
        result = await lead.run(context, cost_tracker)
    except Exception as exc:
        logger.error("data_collection.fatal", error=str(exc))
        run_store.finish(run_id, "failed")
        raise

    summary = {
        "run_id":      run_id,
        "jobs_raw":    len(result.get("raw_jobs",     [])),
        "jobs_new":    len(result.get("deduped_jobs", [])),
        "source_counts": result.get("source_counts", {}),
        "quality":     result.get("collection_quality", "unknown"),
    }
    run_store.finish(run_id, "running", jobs_scraped=summary["jobs_raw"], jobs_new=summary["jobs_new"])
    return summary
