"""
MarketForge AI — Job Source Connector Abstract Base Class.

All job source connectors implement this interface.
Adding a new source = implement one class, register one line in the Lead Agent.
No existing code changes required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from marketforge.models.job import RawJob

from marketforge.nlp.taxonomy import (
    classify_role,
    detect_sponsorship,
    detect_startup,
    extract_salary,
)

logger = structlog.get_logger(__name__)


class JobSourceConnector(ABC):
    """
    Abstract base for all job source connectors.

    Subclasses implement search() and _parse_result().
    The base class provides enrichment (visa, startup, salary NER) and
    a safe_search() wrapper that never crashes the pipeline.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique source identifier: 'adzuna', 'reed', 'wellfound', etc."""
        ...

    @property
    def daily_quota(self) -> int:
        """Max API/scraping calls per day. Override per connector."""
        return 100

    @abstractmethod
    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_per_query: int = 50,
    ) -> list[RawJob]:
        """
        Execute queries and return normalised RawJob objects.
        Subclass is responsible for:
          - HTTP calls / API calls / scraping
          - Parsing source-specific response format into RawJob
          - Respecting daily_quota and rate limits

        The base class handles enrichment (visa, startup, salary NER, role classification).
        """
        ...

    # ── Enrichment pipeline ───────────────────────────────────────────────────

    def enrich(self, job: RawJob) -> RawJob:
        """Apply all enrichment steps to a parsed job."""
        job = self._enrich_salary(job)
        job = self._enrich_sponsorship(job)
        job = self._enrich_startup(job)
        job = self._enrich_role(job)
        return job

    @staticmethod
    def _enrich_salary(job: RawJob) -> RawJob:
        """Extract salary from description when structured fields are missing."""
        if job.salary_min is None and job.salary_max is None and job.description:
            low, high = extract_salary(job.description)
            if low:
                job.salary_min = low
            if high:
                job.salary_max = high
        return job

    @staticmethod
    def _enrich_sponsorship(job: RawJob) -> RawJob:
        if job.offers_sponsorship is None and job.description:
            offers, citizens = detect_sponsorship(job.description)
            job.offers_sponsorship = offers
            job.citizens_only      = citizens
        return job

    @staticmethod
    def _enrich_startup(job: RawJob) -> RawJob:
        if job.description:
            job.is_startup = detect_startup(job.description, job.company)
        return job

    @staticmethod
    def _enrich_role(job: RawJob) -> RawJob:
        if job.role_category is None:
            role, level = classify_role(job.title)
            job.role_category    = role   # type: ignore[assignment]
            job.experience_level = level  # type: ignore[assignment]
        return job

    # ── Safe wrapper ──────────────────────────────────────────────────────────

    async def safe_search(
        self,
        queries: list[str],
        location: str = "UK",
        max_per_query: int = 50,
    ) -> list[RawJob]:
        """
        Search with full error isolation — never raises, never crashes the pipeline.
        Applies enrichment to all returned jobs.
        """
        try:
            jobs = await self.search(queries, location, max_per_query)
            enriched = [self.enrich(job) for job in jobs]
            logger.info(
                "connector.search.done",
                source=self.source_name,
                returned=len(enriched),
                queries=len(queries),
            )
            return enriched
        except Exception as exc:
            logger.error(
                "connector.search.failed",
                source=self.source_name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []
