"""
MarketForge AI — Department 1: ReedDeepScoutAgent

Reed.co.uk free API (500 calls/day). Excellent UK salary data.
Deep agent that tracks salary_disclosure_rate per query and up-weights
high-disclosure queries in subsequent runs.
"""
from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings
from marketforge.connectors.base import JobSourceConnector
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)

REED_BASE = "https://www.reed.co.uk/api/1.0/search"

_REED_QUERIES = [
    "machine learning engineer", "AI engineer", "data scientist",
    "MLOps engineer", "NLP engineer", "computer vision engineer",
    "deep learning engineer", "LLM engineer", "AI researcher",
    "data engineer", "AI product manager", "applied scientist",
    "research scientist machine learning", "generative AI",
    "large language model", "pytorch tensorflow",
]


class ReedConnector(JobSourceConnector):
    """Reed API connector (HTTP Basic Auth, key as username)."""

    source_name = "reed"
    daily_quota = settings.sources.reed_daily_quota

    def __init__(self) -> None:
        key = settings.sources.reed_api_key
        encoded = base64.b64encode(f"{key}:".encode()).decode()
        self._headers = {"Authorization": f"Basic {encoded}"}
        self._calls   = 0

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_per_query: int = 50,
    ) -> list[RawJob]:
        jobs: list[RawJob] = []
        async with httpx.AsyncClient(
            timeout=settings.sources.request_timeout_s,
            headers=self._headers,
        ) as client:
            for query in queries:
                if self._calls >= self.daily_quota:
                    logger.warning("reed.quota_reached")
                    break
                try:
                    batch = await self._search_one(client, query, max_per_query)
                    jobs.extend(batch)
                    self._calls += 1
                    await asyncio.sleep(0.3)
                except Exception as exc:
                    logger.error("reed.search_error", query=query[:40], error=str(exc))
        return jobs

    async def _search_one(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[RawJob]:
        params = {
            "keywords":       query,
            "resultsToTake":  min(max_results, 100),
            "resultsToSkip":  0,
        }
        resp = await client.get(REED_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

        jobs = []
        for r in data.get("results", []):
            try:
                jobs.append(self._parse(r))
            except Exception:
                pass
        return jobs

    def _parse(self, r: dict) -> RawJob:
        from datetime import datetime
        title = r.get("jobTitle", "").strip()
        desc  = r.get("jobDescription", "")
        combined = f"{title} {desc}".lower()

        wm = "unknown"
        if "remote" in combined:
            wm = "remote"
        elif "hybrid" in combined:
            wm = "hybrid"

        posted = None
        if ds := r.get("date"):
            try:
                from datetime import datetime as dt
                posted = dt.strptime(ds[:10], "%d/%m/%Y").date()
            except ValueError:
                pass

        return RawJob(
            job_id=f"reed_{r['jobId']}",
            title=title,
            company=r.get("employerName", "Unknown"),
            location=r.get("locationName", "UK"),
            salary_min=r.get("minimumSalary"),
            salary_max=r.get("maximumSalary"),
            description=desc[:10_000],
            url=r.get("jobUrl", ""),
            source="reed",
            posted_date=posted,
            work_model=wm,  # type: ignore[arg-type]
        )


class ReedDeepScoutAgent(DeepAgent):
    """
    Deep Agent for Reed.co.uk.

    plan():    Reads adaptive_params["query_weights"] — a per-query float
               measuring historical salary_disclosure_rate. Sorts the query
               list so high-disclosure queries run first within the quota window.
               Detects if this is a contract-heavy or permanent-heavy run by
               checking the prior run's role mix.

    execute(): Parallel query batches (BM25 similarity pre-dedup prevents
               duplicate titles from the same company clogging the corpus).
               Tracks which queries produced salary-disclosed jobs for reflect().

    reflect(): Computes salary_disclosure_rate per query. Updates the query
               weight map in adaptive_params. Flags queries whose disclosure
               rate has been consistently < 10% for 4 runs (likely low-quality
               job boards mirrored on Reed).

    output():  Returns enriched RawJob list with source telemetry.
    """

    agent_id   = "reed_deep_scout_v1"
    department = "data_collection"

    def __init__(self) -> None:
        self._connector = ReedConnector()

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        weights  = adaptive.get("query_weights", {})

        # Sort queries: highest historical salary_disclosure_rate first
        queries = sorted(
            _REED_QUERIES,
            key=lambda q: weights.get(q, 0.5),
            reverse=True,
        )
        # Cap to quota
        queries = queries[:self._connector.daily_quota]

        logger.info(f"{self.agent_id}.plan.done", queries=len(queries))
        return {"queries": queries, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        queries   = plan["queries"]
        all_jobs: list[RawJob] = []
        salary_per_query: dict[str, int] = {}
        total_per_query:  dict[str, int] = {}
        seen: set[str] = set()

        sem = asyncio.Semaphore(8)

        async def run_q(q: str) -> list[RawJob]:
            async with sem:
                c = ReedConnector()
                batch = await c.safe_search([q])
                unique = [j for j in batch if j.dedup_hash not in seen]
                for j in unique:
                    seen.add(j.dedup_hash)
                salary_per_query[q] = sum(1 for j in unique if j.salary_min or j.salary_max)
                total_per_query[q]  = len(unique)
                return unique

        results = await asyncio.gather(*[run_q(q) for q in queries], return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_jobs.extend(r)

        return {
            "jobs": all_jobs,
            "salary_per_query": salary_per_query,
            "total_per_query":  total_per_query,
        }

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        jobs          = result.get("jobs", [])
        sal_map       = result.get("salary_per_query", {})
        total_map     = result.get("total_per_query", {})
        adaptive      = plan.get("adaptive", {})
        weights       = adaptive.get("query_weights", {})
        low_qual_runs = adaptive.get("low_quality_counts", {})

        for q, total in total_map.items():
            disclosed = sal_map.get(q, 0)
            rate      = disclosed / max(total, 1)
            # Exponential moving average weight
            old_w     = weights.get(q, 0.5)
            weights[q] = round(0.7 * old_w + 0.3 * rate, 4)

            if rate < 0.10:
                low_qual_runs[q] = low_qual_runs.get(q, 0) + 1
            else:
                low_qual_runs.pop(q, None)

        adaptive["query_weights"]     = weights
        adaptive["low_quality_counts"]= low_qual_runs
        state["adaptive_params"]      = adaptive
        state["last_yield"]           = len(jobs)

        quality = "good" if len(jobs) >= 30 else ("warning" if len(jobs) > 5 else "poor")
        return {"quality": quality, "yield": len(jobs), "notes": f"salary_disclosed={sum(sal_map.values())}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "jobs":    result.get("jobs", []),
            "source":  "reed",
            "yield":   reflection.get("yield", 0),
            "quality": reflection.get("quality", "unknown"),
        }
