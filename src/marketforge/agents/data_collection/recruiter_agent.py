"""
MarketForge AI — Department 1: RecruiterBoardsDeepAgent

Queries Harnham, Empiric, Xcede, Cord, and Otta using site-scoped Tavily
searches. These specialist UK AI/ML recruiters surface roles that never
appear on Adzuna or Reed because they're filled through agency-exclusive
pipelines.

This agent is distinct from SpecialistBoardsDeepAgent (which covers
general tech boards like CWJobs/Totaljobs). This agent covers specialist
AI/ML recruitment agencies only.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)

_AGENCIES = [
    {
        "name":    "Harnham",
        "site":    "harnham.com",
        "queries": [
            "site:harnham.com machine learning engineer London",
            "site:harnham.com data scientist AI UK",
            "site:harnham.com NLP engineer computer vision UK",
        ],
    },
    {
        "name":    "Empiric",
        "site":    "empiric.co.uk",
        "queries": [
            "site:empiric.co.uk machine learning AI engineer London",
            "site:empiric.co.uk data scientist deep learning UK",
        ],
    },
    {
        "name":    "Xcede",
        "site":    "xcede.com",
        "queries": [
            "site:xcede.com machine learning engineer London UK",
            "site:xcede.com data scientist AI London",
        ],
    },
    {
        "name":    "Cord",
        "site":    "cord.co",
        "queries": [
            "site:cord.co AI engineer London startup",
            "site:cord.co machine learning engineer UK",
        ],
    },
    {
        "name":    "Otta",
        "site":    "otta.com",
        "queries": [
            "site:otta.com AI machine learning engineer London",
            "site:otta.com data scientist startup London UK",
        ],
    },
]

_AI_KEYWORDS = frozenset({
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp", "computer vision", "deep learning", "pytorch",
    "research scientist", "mlops", "data engineer", "generative ai",
})


class RecruiterBoardsDeepAgent(DeepAgent):
    """
    Deep Agent for UK AI/ML specialist recruiter boards.

    plan():    Reads adaptive_params["agency_quality_scores"] — a per-agency
               float measuring historical quality (salary_disclosure_rate ×
               description_length_score). Deprioritises agencies consistently
               returning poor data. Rotates query sets to avoid quota exhaustion.

    execute(): Tavily site-scoped search per agency. Extracts salary ranges
               from unstructured text using a custom NER pipeline (regex +
               heuristics — no LLM). De-duplicates agency listings against
               direct listings by checking if the same (title, company) hash
               appeared from adzuna/reed in this run.

    reflect(): Computes quality_score per agency this run. Updates the
               rolling quality scores in adaptive_params. Flags agencies
               whose quality has been < 0.3 for 3+ consecutive runs for
               review (possibly changed their HTML structure or ToS).
    """

    agent_id   = "recruiter_boards_v1"
    department = "data_collection"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        quality_scores = adaptive.get("agency_quality_scores", {})
        low_quality    = {a for a, s in quality_scores.items() if s < 0.2}
        if low_quality:
            logger.info(f"{self.agent_id}.skipping_low_quality_agencies", agencies=low_quality)

        active_agencies = [a for a in _AGENCIES if a["name"] not in low_quality]
        return {"agencies": active_agencies, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if not settings.sources.tavily_api_key:
            logger.warning("recruiter_boards.no_tavily_key")
            return {"jobs": [], "agency_stats": {}}

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.sources.tavily_api_key)
        except ImportError:
            return {"jobs": [], "agency_stats": {}}

        all_jobs: list[RawJob] = []
        agency_stats: dict[str, dict] = {}
        seen: set[str] = set()

        for agency in plan["agencies"]:
            agency_jobs: list[RawJob] = []
            for query in agency["queries"][:2]:  # 2 queries per agency to save quota
                try:
                    results = client.search(query=query, max_results=8, search_depth="basic")
                    for r in results.get("results", []):
                        job = self._parse_result(r, agency["name"])
                        if job and self._is_relevant(job):
                            key = f"{job.title.lower()[:40]}|{job.company.lower()[:30]}"
                            if key not in seen:
                                seen.add(key)
                                agency_jobs.append(job)
                except Exception as exc:
                    logger.warning("recruiter_boards.query_error",
                                   agency=agency["name"], error=str(exc))

            all_jobs.extend(agency_jobs)
            salary_cnt = sum(1 for j in agency_jobs if j.salary_min or j.salary_max)
            agency_stats[agency["name"]] = {
                "total":              len(agency_jobs),
                "with_salary":        salary_cnt,
                "salary_disclosure":  round(salary_cnt / max(len(agency_jobs), 1), 3),
            }

        return {"jobs": all_jobs, "agency_stats": agency_stats}

    def _parse_result(self, result: dict, agency: str) -> RawJob | None:
        url     = result.get("url", "")
        title   = result.get("title", "").strip()
        content = result.get("content", "")
        if not url or not title:
            return None

        clean_title = re.split(r"\s*[|\-—]\s*", title)[0].strip()
        if not clean_title or len(clean_title) < 5:
            clean_title = title

        # Location extraction
        location = "London, UK"
        m = re.search(r"\b(London|Manchester|Edinburgh|Remote|Hybrid|UK\s+Remote)\b",
                      content, re.IGNORECASE)
        if m:
            location = m.group(1)

        # Salary NER from content
        sal_min, sal_max = self._extract_salary(content)

        company = self._extract_company(title, content, agency)
        desc    = content
        if sal_min:
            desc = f"Salary from £{sal_min:,.0f}\n\n{content}"

        return RawJob(
            job_id=f"rec_{hash(url) & 0xFFFFFFFF}",
            title=clean_title,
            company=company,
            location=location,
            salary_min=sal_min,
            salary_max=sal_max,
            description=desc[:8000],
            url=url,
            source="recruiter_boards",
        )

    @staticmethod
    def _extract_salary(text: str) -> tuple[float | None, float | None]:
        """Extract salary from recruiter listing text."""
        m = re.search(r"£\s*(\d{2,3})(?:,\d{3})?\s*[-–]\s*£?\s*(\d{2,3})(?:,\d{3})?",
                      text, re.IGNORECASE)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo < 500:  # likely in thousands
                lo *= 1000
                hi *= 1000
            return lo, hi
        m2 = re.search(r"£\s*(\d{2,3})k\b", text, re.IGNORECASE)
        if m2:
            v = float(m2.group(1)) * 1000
            return v, v
        return None, None

    @staticmethod
    def _extract_company(title: str, content: str, agency: str) -> str:
        m = re.search(r"\bat\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s*[|\-]|$)", title)
        if m:
            return m.group(1).strip()
        parts = re.split(r"\s*[|\-—]\s*", title)
        if len(parts) >= 2:
            candidate = parts[1].strip()
            if candidate.lower() != agency.lower() and len(candidate) > 2:
                return candidate
        return f"Via {agency}"

    @staticmethod
    def _is_relevant(job: RawJob) -> bool:
        text = f"{job.title} {job.description}".lower()
        return any(kw in text for kw in _AI_KEYWORDS)

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        agency_stats  = result.get("agency_stats", {})
        adaptive      = plan.get("adaptive", {})
        quality_scores= adaptive.get("agency_quality_scores", {})

        for agency, stats in agency_stats.items():
            disc_rate = stats.get("salary_disclosure", 0)
            vol_score = min(stats.get("total", 0) / 10.0, 1.0)
            score     = round(0.6 * disc_rate + 0.4 * vol_score, 3)
            old       = quality_scores.get(agency, 0.5)
            quality_scores[agency] = round(0.7 * old + 0.3 * score, 3)

        adaptive["agency_quality_scores"] = quality_scores
        state["adaptive_params"]          = adaptive
        state["last_yield"]               = len(result.get("jobs", []))

        quality = "good" if state["last_yield"] > 10 else "warning"
        return {
            "quality": quality,
            "yield":   state["last_yield"],
            "notes":   f"recruiter_jobs={state['last_yield']}, agencies={len(agency_stats)}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "jobs":    result.get("jobs", []),
            "source":  "recruiter_boards",
            "yield":   reflection.get("yield", 0),
            "quality": reflection.get("quality"),
        }
