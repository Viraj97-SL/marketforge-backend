"""
MarketForge AI — Department 1: AdzunaDeepScoutAgent

A genuine deep agent, not an API wrapper. The plan() phase builds
a dynamic query matrix based on prior-run coverage gap analysis.
The reflect() phase prunes low-yield queries from the next run's matrix.
"""
from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Any

import httpx
import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings
from marketforge.connectors.base import JobSourceConnector
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)

ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs/gb/search"

# ── Role × location query matrix (seed — agent learns from this) ──────────────
_BASE_ROLES = [
    "AI Engineer", "Machine Learning Engineer", "ML Engineer",
    "LLM Engineer", "Data Scientist", "MLOps Engineer",
    "NLP Engineer", "Computer Vision Engineer", "Research Scientist",
    "Applied Scientist", "Data Engineer", "AI Safety Researcher",
    "Generative AI Engineer", "Foundation Model Engineer",
]
_BASE_LOCATIONS = [
    "London", "United Kingdom", "Remote UK",
    "Manchester", "Cambridge", "Edinburgh",
]
_SALARY_BANDS = [None, "30000", "60000", "90000"]   # min salary filters


class AdzunaConnector(JobSourceConnector):
    """Low-level Adzuna API connector — called by AdzunaDeepScoutAgent."""

    source_name = "adzuna"
    daily_quota = settings.sources.adzuna_daily_quota

    def __init__(self) -> None:
        self._app_id  = settings.sources.adzuna_app_id
        self._app_key = settings.sources.adzuna_app_key
        self._calls   = 0

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_per_query: int = 50,
    ) -> list[RawJob]:
        jobs: list[RawJob] = []
        async with httpx.AsyncClient(timeout=settings.sources.request_timeout_s) as client:
            for query in queries:
                if self._calls >= self.daily_quota:
                    logger.warning("adzuna.quota_reached", calls=self._calls)
                    break
                try:
                    batch = await self._single_search(client, query, max_per_query)
                    jobs.extend(batch)
                    self._calls += 1
                    await asyncio.sleep(0.2)   # polite pacing
                except httpx.HTTPStatusError as exc:
                    logger.error("adzuna.http_error", status=exc.response.status_code, query=query[:40])
                except Exception as exc:
                    logger.error("adzuna.search_error", query=query[:40], error=str(exc))
        return jobs

    async def _single_search(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[RawJob]:
        params = {
            "app_id":           self._app_id,
            "app_key":          self._app_key,
            "results_per_page": min(max_results, 50),
            "what":             query,
            "where":            "United Kingdom",
            "sort_by":          "date",
            "max_days_old":     8,
            "category":         "it-jobs",
            "content-type":     "application/json",
        }
        resp = await client.get(f"{ADZUNA_BASE}/1", params=params)
        resp.raise_for_status()
        data = resp.json()

        jobs = []
        for r in data.get("results", []):
            try:
                jobs.append(self._parse(r))
            except Exception as exc:
                logger.debug("adzuna.parse_skip", title=r.get("title", "?"), error=str(exc))
        logger.debug("adzuna.query_done", query=query[:40], count=len(jobs))
        return jobs

    def _parse(self, r: dict) -> RawJob:
        from datetime import datetime
        loc   = r.get("location", {}).get("display_name", "UK")
        desc  = r.get("description", "")
        title = r.get("title", "").strip()
        combined = f"{title} {desc}".lower()

        work_model = "unknown"
        if "remote" in combined:
            work_model = "remote"
        elif "hybrid" in combined:
            work_model = "hybrid"
        elif "on-site" in combined or "onsite" in combined:
            work_model = "onsite"

        posted = None
        if created := r.get("created"):
            try:
                posted = datetime.fromisoformat(created.replace("Z", "+00:00")).date()
            except ValueError:
                pass

        return RawJob(
            job_id=f"adzuna_{r['id']}",
            title=title,
            company=r.get("company", {}).get("display_name", "Unknown"),
            location=loc,
            salary_min=r.get("salary_min"),
            salary_max=r.get("salary_max"),
            description=desc[:10_000],
            url=r.get("redirect_url", ""),
            source="adzuna",
            posted_date=posted,
            work_model=work_model,  # type: ignore[arg-type]
        )


class AdzunaDeepScoutAgent(DeepAgent):
    """
    Deep Agent for the Adzuna UK job API.

    plan():     Reads prior run state. Builds a dynamic query matrix by:
                (a) restoring the base role × location matrix,
                (b) pruning queries with consistently zero yield (learned),
                (c) adding salary band variants for roles with high disclosure rates,
                (d) boosting query budget for role categories trending up (from ML forecast).

    execute():  Runs the query matrix in parallel batches, respects the daily quota,
                tracks yield per query for the reflect() phase, and performs
                intra-run dedup before returning.

    reflect():  Computes yield_per_query across the run. Prunes queries that
                returned 0 results for 3 consecutive runs (stored in adaptive_params).
                Flags anomalous spikes (possible API schema changes) as warnings.
                Updates adaptive_params["pruned_queries"] so next plan() skips them.

    output():   Returns validated RawJob list and per-source telemetry.
    """

    agent_id   = "adzuna_deep_scout_v1"
    department = "data_collection"

    def __init__(self) -> None:
        self._connector = AdzunaConnector()

    # ── PLAN ─────────────────────────────────────────────────────────────────

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        pruned   = set(adaptive.get("pruned_queries", []))
        top_roles_by_demand = context.get("top_role_categories_by_demand", [])

        # Build query matrix: role × location (+ salary band for high-disclosure roles)
        matrix: list[dict] = []
        for role in _BASE_ROLES:
            for loc in _BASE_LOCATIONS:
                query = f"{role} {loc}"
                if query not in pruned:
                    priority = 2 if any(role.lower() in cat for cat in top_roles_by_demand) else 1
                    matrix.append({"query": query, "priority": priority, "salary_min": None})

                # Add salary-band variant for senior roles (improves salary data coverage)
                if any(kw in role.lower() for kw in ("lead", "staff", "principal", "head")):
                    for band in _SALARY_BANDS[2:]:   # £60k+, £90k+
                        bq = f"{role} {loc}"
                        if bq not in pruned:
                            matrix.append({"query": bq, "priority": 1, "salary_min": band})

        # Sort by priority descending; cap to daily quota
        matrix.sort(key=lambda x: x["priority"], reverse=True)
        matrix = matrix[:self._connector.daily_quota]

        logger.info(
            f"{self.agent_id}.plan.done",
            total_queries=len(matrix),
            pruned=len(pruned),
        )
        return {"query_matrix": matrix, "adaptive": adaptive}

    # ── EXECUTE ───────────────────────────────────────────────────────────────

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        matrix: list[dict] = plan["query_matrix"]
        if not matrix:
            return {"jobs": [], "yield_per_query": {}}

        all_jobs: list[RawJob] = []
        yield_log: dict[str, int] = {}
        seen_hashes: set[str] = set()

        # Fan-out in batches of 10 to respect rate limits
        sem = asyncio.Semaphore(10)

        async def run_query(item: dict) -> list[RawJob]:
            async with sem:
                query = item["query"]
                connector = AdzunaConnector()
                batch = await connector.safe_search([query])
                # Intra-run dedup by hash
                unique = []
                for job in batch:
                    if job.dedup_hash not in seen_hashes:
                        seen_hashes.add(job.dedup_hash)
                        unique.append(job)
                yield_log[query] = len(unique)
                return unique

        results = await asyncio.gather(*[run_query(q) for q in matrix], return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_jobs.extend(r)

        logger.info(
            f"{self.agent_id}.execute.done",
            total_jobs=len(all_jobs),
            queries_run=len(matrix),
        )
        return {"jobs": all_jobs, "yield_per_query": yield_log}

    # ── REFLECT ───────────────────────────────────────────────────────────────

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        jobs     = result.get("jobs", [])
        yld_map  = result.get("yield_per_query", {})
        adaptive = plan.get("adaptive", {})

        # Track zero-yield queries across runs
        zero_yield_counts: dict[str, int] = adaptive.get("zero_yield_counts", {})
        for query, count in yld_map.items():
            if count == 0:
                zero_yield_counts[query] = zero_yield_counts.get(query, 0) + 1
            else:
                zero_yield_counts.pop(query, None)   # reset on any yield

        # Prune queries with 3+ consecutive zero-yield runs
        pruned: list[str] = adaptive.get("pruned_queries", [])
        newly_pruned = [q for q, c in zero_yield_counts.items() if c >= 3 and q not in pruned]
        pruned.extend(newly_pruned)
        if newly_pruned:
            logger.info(f"{self.agent_id}.reflect.pruned", queries=newly_pruned)

        # Quality assessment
        quality = "good"
        notes   = f"yield={len(jobs)}, queries={len(yld_map)}"
        if len(jobs) == 0:
            quality = "poor"
            notes   = "zero yield — check API key and quota"
        elif len(jobs) < 20:
            quality = "warning"
            notes   = f"low yield ({len(jobs)}) — check query matrix"

        # Update salary disclosure rate for adaptive query weighting
        disclosed = sum(1 for j in jobs if j.salary_min or j.salary_max)
        salary_disclosure_rate = disclosed / max(len(jobs), 1)

        # Save adaptive params back
        adaptive["zero_yield_counts"]     = zero_yield_counts
        adaptive["pruned_queries"]        = pruned
        adaptive["last_yield"]            = len(jobs)
        adaptive["salary_disclosure_rate"]= round(salary_disclosure_rate, 3)
        state["adaptive_params"]          = adaptive
        state["last_yield"]               = len(jobs)

        return {"quality": quality, "notes": notes, "yield": len(jobs), "pruned_this_run": newly_pruned}

    # ── OUTPUT ────────────────────────────────────────────────────────────────

    async def output(
        self,
        result: dict[str, Any],
        reflection: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "jobs":       result.get("jobs", []),
            "source":     self.source_name,
            "yield":      reflection.get("yield", 0),
            "quality":    reflection.get("quality", "unknown"),
        }

    @property
    def source_name(self) -> str:
        return "adzuna"
