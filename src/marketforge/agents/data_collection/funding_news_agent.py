"""
MarketForge AI — Department 1: FundingNewsDeepDiscoveryAgent

Monitors Sifted, TechCrunch UK, Beauhurst, and City A.M. via Tavily to
discover newly funded UK AI companies before they appear on job boards.
Newly funded startups almost always start hiring within weeks of a round.

This agent feeds two outputs:
  1. Direct RawJob objects from careers pages it crawls
  2. discovered_companies list → injected into CareerPagesDeepCrawlerAgent
     watchlist for this run
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)

# News search queries targeting UK AI funding
_FUNDING_QUERIES = [
    "UK AI startup funding round seed series London 2025 2026",
    "London machine learning startup raised investment 2026",
    "UK deep tech AI series A series B funding announced",
    "British AI company seed round artificial intelligence funding",
    "Sifted UK AI startup funding round 2026",
    "TechCrunch UK AI machine learning startup raises 2026",
]

# Domains to skip (news sites, social media, etc.)
_SKIP_DOMAINS = {
    "techcrunch.com", "sifted.eu", "twitter.com", "x.com", "linkedin.com",
    "crunchbase.com", "bloomberg.com", "reuters.com", "ft.com",
    "theguardian.com", "bbc.co.uk", "google.com", "youtube.com",
    "facebook.com", "instagram.com", "medium.com", "substack.com",
    "github.com", "notion.so", "docs.google.com", "angel.co",
    "wellfound.com", "glassdoor.com", "indeed.com", "adzuna.co.uk",
    "pitchbook.com", "beauhurst.com", "businessinsider.com",
}

# Career page paths to probe
_CAREER_PATHS = [
    "/careers", "/jobs", "/work-with-us", "/join", "/join-us",
    "/about/careers", "/company/jobs", "/hiring", "/open-positions",
]

_AI_KEYWORDS = frozenset({
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp", "computer vision", "deep learning", "pytorch",
    "research scientist", "applied scientist", "generative ai",
    "mlops", "data engineer", "langchain", "multimodal",
})


class FundingNewsDeepDiscoveryAgent(DeepAgent):
    """
    Deep Agent for discovering newly funded UK AI startups.

    plan():    Reads adaptive_params["seen_companies"] to avoid re-crawling
               companies already in the watchlist. Selects which news queries
               to run based on prior coverage (rotates query set each run to
               maximise coverage while staying within Tavily quota).

    execute(): Step 1 — Tavily news search across _FUNDING_QUERIES.
               Step 2 — Extract company domains from search results.
               Step 3 — Score each company: funding_amount_signal ×
               ai_relevance × uk_presence (0-1 composite).
               Step 4 — Probe careers pages of highest-scored companies.
               Step 5 — Extract AI/ML job listings via JSON-LD or link scan.

    reflect(): Checks how many new companies were discovered vs. the prior
               4-week average. A sudden drop (< 2 new companies) may indicate
               the news queries are stale — flags for query refresh.
               Updates seen_companies to prevent re-crawling.
    """

    agent_id   = "funding_news_discovery_v1"
    department = "data_collection"

    # Published after execute() for CareerPagesDeepCrawlerAgent to consume
    discovered_companies: list[dict] = []

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive     = state.get("adaptive_params", {})
        seen         = set(adaptive.get("seen_company_domains", []))
        query_offset = adaptive.get("query_offset", 0)
        # Rotate through queries across runs for coverage breadth
        rotated = _FUNDING_QUERIES[query_offset:] + _FUNDING_QUERIES[:query_offset]
        next_offset = (query_offset + 3) % len(_FUNDING_QUERIES)
        adaptive["query_offset"] = next_offset
        return {"queries": rotated[:5], "seen_domains": seen, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if not settings.sources.tavily_api_key:
            logger.warning("funding_news.no_tavily_key")
            return {"jobs": [], "new_companies": []}

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.sources.tavily_api_key)
        except ImportError:
            logger.warning("funding_news.tavily_not_installed")
            return {"jobs": [], "new_companies": []}

        seen      = plan["seen_domains"]
        queries   = plan["queries"]

        # Step 1+2: Mine funding news for company domains
        candidate_domains: dict[str, dict] = {}
        for query in queries:
            try:
                results = client.search(query=query, max_results=8,
                                        search_depth="advanced", topic="news")
                for r in results.get("results", []):
                    for company in self._extract_companies(r):
                        d = company["domain"]
                        if d not in seen and d not in candidate_domains:
                            candidate_domains[d] = company
                await asyncio.sleep(0.5)
            except Exception as exc:
                logger.warning("funding_news.query_error", query=query[:40], error=str(exc))

        logger.info("funding_news.candidates", count=len(candidate_domains))

        if not candidate_domains:
            return {"jobs": [], "new_companies": []}

        # Step 3: Score companies by AI relevance
        scored = [
            (self._score_company(c), c)
            for c in candidate_domains.values()
        ]
        scored.sort(key=lambda x: -x[0])
        top_candidates = [c for _, c in scored[:20]]  # probe top 20

        # Step 4+5: Probe careers pages
        all_jobs: list[RawJob] = []
        new_companies: list[dict] = []

        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MarketForgeBot/0.1)"},
        ) as http:
            tasks = [self._probe_careers(http, c) for c in top_candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for company, result in zip(top_candidates, results):
            if isinstance(result, list) and result:
                all_jobs.extend(result)
                new_companies.append({
                    "company":     company["company"],
                    "careers_url": company.get("careers_url", company["website"]),
                    "stage":       "recently_funded",
                })

        # Expose for CareerPagesDeepCrawlerAgent
        FundingNewsDeepDiscoveryAgent.discovered_companies = new_companies

        return {"jobs": all_jobs, "new_companies": new_companies}

    def _extract_companies(self, result: dict) -> list[dict]:
        companies = []
        text = f"{result.get('title','')} {result.get('content','')}"
        for match in re.finditer(r'https?://(?:www\.)?([a-z0-9.\-]+)', text.lower()):
            raw_domain = match.group(1).strip(".")
            if any(skip in raw_domain for skip in _SKIP_DOMAINS):
                continue
            if re.search(r'\.(ai|io|co\.uk|com)$', raw_domain):
                parts = raw_domain.split(".")
                if len(parts[0]) > 2:
                    name = parts[0].replace("-", " ").title()
                    companies.append({
                        "company":  name,
                        "domain":   raw_domain,
                        "website":  f"https://{raw_domain}",
                        "careers_url": f"https://{raw_domain}/careers",
                    })
        return companies[:4]

    def _score_company(self, company: dict) -> float:
        """Score 0-1 based on AI relevance signals in company name/domain."""
        ai_signals = ["ai", "ml", "data", "intelligence", "deep", "neural",
                      "learn", "vision", "nlp", "model", "robot", "auto"]
        name = company.get("company", "").lower()
        domain = company.get("domain", "").lower()
        hits = sum(1 for s in ai_signals if s in name or s in domain)
        return min(hits / 3.0, 1.0)

    async def _probe_careers(
        self, client: httpx.AsyncClient, company: dict
    ) -> list[RawJob]:
        """Probe career page paths for the company, return AI/ML jobs found."""
        website = company["website"]
        for path in _CAREER_PATHS:
            url = f"{website.rstrip('/')}{path}"
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and len(resp.content) > 300:
                    jobs = self._extract_jobs(resp.text, url, company["company"])
                    if jobs:
                        company["careers_url"] = url
                        return jobs
            except Exception:
                continue
        return []

    def _extract_jobs(self, html: str, page_url: str, company: str) -> list[RawJob]:
        """Extract AI/ML job listings from careers page HTML."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        jobs: list[RawJob] = []
        seen: set[str] = set()

        # JSON-LD first
        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data  = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("JobPosting", "jobPosting"):
                        title = item.get("title", "")
                        if any(kw in title.lower() for kw in _AI_KEYWORDS):
                            jobs.append(RawJob(
                                job_id=f"fn_{hash(page_url + title) & 0xFFFFFFFF}",
                                title=title, company=company, location="UK",
                                description=(item.get("description","") or "")[:8000],
                                url=item.get("url", page_url),
                                source="funding_news", is_startup=True,
                            ))
            except Exception:
                pass

        if jobs:
            return jobs

        # Link scan fallback
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if not text or len(text) < 5 or len(text) > 100:
                continue
            if href.startswith("/"):
                href = urljoin(page_url, href)
            if href in seen:
                continue
            if any(kw in text.lower() for kw in _AI_KEYWORDS):
                seen.add(href)
                jobs.append(RawJob(
                    job_id=f"fn_{hash(href) & 0xFFFFFFFF}",
                    title=text.strip(), company=company, location="UK",
                    description=f"{text} at {company} — discovered via funding news.",
                    url=href, source="funding_news", is_startup=True,
                ))
        return jobs[:10]

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        new_companies = result.get("new_companies", [])
        adaptive      = plan.get("adaptive", {})

        seen = list(adaptive.get("seen_company_domains", []))
        seen.extend(c.get("careers_url", "")[:60] for c in new_companies)
        adaptive["seen_company_domains"] = seen[-500:]

        history = adaptive.get("weekly_discovery_counts", [])
        history.append(len(new_companies))
        adaptive["weekly_discovery_counts"] = history[-4:]

        avg = sum(history) / max(len(history), 1)
        quality = "good"
        if avg < 1.5 and len(history) >= 3:
            quality = "warning"
            logger.warning(f"{self.agent_id}.low_discovery_rate", avg=avg)

        state["adaptive_params"] = adaptive
        state["last_yield"]      = len(result.get("jobs", []))
        return {
            "quality":     quality,
            "new_companies": len(new_companies),
            "yield":       len(result.get("jobs", [])),
            "notes":       f"new_companies={len(new_companies)}, jobs={state['last_yield']}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "jobs":             result.get("jobs", []),
            "source":           "funding_news",
            "new_companies":    result.get("new_companies", []),
            "yield":            reflection.get("yield", 0),
            "quality":          reflection.get("quality"),
        }
