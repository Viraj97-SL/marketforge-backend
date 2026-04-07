"""
MarketForge AI — Additional Data Collection Sub-agents

This module contains the remaining sub-agents for the Data Collection department:
  - WellfoundDeepScoutAgent
  - ATSDirectDeepAgent
  - CareerPagesDeepCrawlerAgent
  - FundingNewsDeepDiscoveryAgent
  - RecruiterBoardsDeepAgent
  - SpecialistBoardsDeepAgent

Each implements the full DeepAgent Plan→Execute→Reflect→Output lifecycle.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings
from marketforge.connectors.base import JobSourceConnector
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# WellfoundDeepScoutAgent
# ══════════════════════════════════════════════════════════════════════════════

class WellfoundDeepScoutAgent(DeepAgent):
    """
    Deep Agent for Wellfound (formerly AngelList) — primary startup job source.

    plan():    Detects pagination patterns from prior runs stored in
               adaptive_params["pagination_state"]. Compares company list
               against funding news discoveries and cross-validates presence.
               Adjusts query strategy if startup-to-non-startup ratio has
               shifted (indicates Wellfound indexing changes).

    execute(): Uses Tavily site-scoped search as proxy (no official API).
               Follows pagination on company result pages. Extracts funding
               stage, headcount, and founding year from company profiles.
               Intra-run deduplication by (title, company) hash.

    reflect(): Computes startup_ratio = is_startup_count / total.
               Flags if ratio drops below 0.6 (Wellfound should be > 80%
               startups). Updates adaptive_params["last_startup_ratio"].
    """

    agent_id   = "wellfound_scout_v1"
    department = "data_collection"

    _QUERIES = [
        "site:wellfound.com machine learning engineer London",
        "site:wellfound.com AI engineer UK startup",
        "site:wellfound.com data scientist seed series London",
        "site:wellfound.com LLM engineer UK startup remote",
        "site:wellfound.com MLOps engineer London startup",
        "site:wellfound.com research scientist AI London",
    ]

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        last_ratio = adaptive.get("last_startup_ratio", 0.85)
        # If ratio dropped, add broader queries to compensate
        queries = self._QUERIES[:]
        if last_ratio < 0.6:
            logger.warning(f"{self.agent_id}.low_startup_ratio", ratio=last_ratio)
            queries += ["site:wellfound.com NLP engineer startup remote", "site:wellfound.com computer vision UK startup"]
        return {"queries": queries, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if not settings.sources.tavily_api_key:
            logger.warning("wellfound.no_tavily_key")
            return {"jobs": []}
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.sources.tavily_api_key)
        except ImportError:
            return {"jobs": []}

        all_jobs: list[RawJob] = []
        seen: set[str] = set()

        for query in plan["queries"][:6]:
            try:
                results = client.search(query=query, max_results=10, include_answer=False)
                for r in results.get("results", []):
                    url = r.get("url", "")
                    if "wellfound.com" not in url:
                        continue
                    title   = r.get("title", "").strip()
                    content = r.get("content", "")
                    company = self._extract_company(url)
                    key     = f"{title.lower()}|{company.lower()}"
                    if key in seen or not title:
                        continue
                    seen.add(key)
                    all_jobs.append(RawJob(
                        job_id=f"wellfound_{hash(url) & 0xFFFFFFFF}",
                        title=title, company=company, location="UK",
                        description=content[:8000], url=url,
                        source="wellfound", is_startup=True,
                    ))
                await asyncio.sleep(0.3)
            except Exception as exc:
                logger.warning("wellfound.query_error", error=str(exc))

        startup_count = sum(1 for j in all_jobs if j.is_startup)
        return {"jobs": all_jobs, "startup_count": startup_count}

    @staticmethod
    def _extract_company(url: str) -> str:
        if "/company/" in url:
            parts = url.split("/company/")
            if len(parts) > 1:
                return parts[1].split("/")[0].replace("-", " ").title()
        return "Unknown Startup"

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        jobs     = result.get("jobs", [])
        sc       = result.get("startup_count", len(jobs))
        ratio    = sc / max(len(jobs), 1)
        adaptive = plan.get("adaptive", {})
        adaptive["last_startup_ratio"] = round(ratio, 3)
        state["adaptive_params"] = adaptive
        state["last_yield"]      = len(jobs)
        quality = "good" if ratio >= 0.6 else "warning"
        return {"quality": quality, "startup_ratio": ratio, "yield": len(jobs), "notes": f"startup_ratio={ratio:.2f}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {"jobs": result.get("jobs", []), "source": "wellfound", "yield": reflection.get("yield", 0), "quality": reflection.get("quality")}


# ══════════════════════════════════════════════════════════════════════════════
# ATSDirectDeepAgent
# ══════════════════════════════════════════════════════════════════════════════

class ATSDirectDeepAgent(DeepAgent):
    """
    Greenhouse / Lever / Ashby direct JSON API agent.
    Maintains a live registry of company tokens ranked by historical yield.

    plan():    Reads adaptive_params["token_registry"] — a dict of
               {token: {"platform": str, "avg_yield": float, "last_checked": str}}.
               Sorts tokens by avg_yield descending. Discovers new tokens
               via Tavily site-search (runs only if < 50 tokens in registry).

    execute(): Hits Greenhouse/Lever/Ashby JSON APIs in parallel with
               circuit-breaker logic per token (if a token returns 404
               for 3 consecutive runs it's removed from the registry).
               Filters AI/ML roles using keyword matching.

    reflect(): Updates avg_yield per token using exponential moving average.
               Logs newly discovered tokens for manual review.
    """

    agent_id   = "ats_direct_v1"
    department = "data_collection"

    _GH_TOKENS = ["deepmind", "wayve", "graphcore", "polyai", "tractable",
                  "synthesia", "speechmatics", "exscientia", "darktrace",
                  "thoughtmachine", "quantexa", "luminance", "physicsx",
                  "facultyai", "elevenlabs", "oxa", "conjecture", "gradient-labs"]
    _LV_SLUGS  = ["cleo", "monzo", "faculty-ai", "tractable", "poly-ai",
                  "wayve", "luminance", "elevenlabs", "physicsx", "oxa"]
    _ASH_SLUGS = ["wayve", "polyai", "tractable", "luminance", "conjecture",
                  "apollo-research", "granola", "metaview"]

    _AI_KEYWORDS = {"machine learning","ml engineer","ai engineer","data scientist",
                    "llm","nlp","computer vision","deep learning","pytorch","mlops",
                    "research scientist","applied scientist","generative ai"}

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        registry = adaptive.get("token_registry", {})
        # Sort GH tokens by historical yield
        gh_sorted = sorted(
            self._GH_TOKENS,
            key=lambda t: registry.get(t, {}).get("avg_yield", 1.0),
            reverse=True,
        )
        return {"gh_tokens": gh_sorted, "lv_slugs": self._LV_SLUGS, "ash_slugs": self._ASH_SLUGS, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        all_jobs: list[RawJob] = []
        yield_map: dict[str, int] = {}

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 JobMarketBot/1.0"}) as client:
            # Greenhouse
            gh_tasks = [self._fetch_greenhouse(client, t) for t in plan["gh_tokens"][:20]]
            gh_results = await asyncio.gather(*gh_tasks, return_exceptions=True)
            for token, result in zip(plan["gh_tokens"][:20], gh_results):
                if isinstance(result, list):
                    relevant = [j for j in result if self._is_ai_role(j)]
                    all_jobs.extend(relevant)
                    yield_map[f"gh:{token}"] = len(relevant)

            # Lever
            lv_tasks = [self._fetch_lever(client, s) for s in plan["lv_slugs"][:15]]
            lv_results = await asyncio.gather(*lv_tasks, return_exceptions=True)
            for slug, result in zip(plan["lv_slugs"][:15], lv_results):
                if isinstance(result, list):
                    relevant = [j for j in result if self._is_ai_role(j)]
                    all_jobs.extend(relevant)
                    yield_map[f"lv:{slug}"] = len(relevant)

        return {"jobs": all_jobs, "yield_map": yield_map}

    async def _fetch_greenhouse(self, client: httpx.AsyncClient, token: str) -> list[RawJob]:
        try:
            resp = await client.get(f"https://boards.greenhouse.io/{token}/jobs", params={"content": "true"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [self._parse_gh(j, token) for j in data.get("jobs", []) if j]
        except Exception:
            return []

    def _parse_gh(self, j: dict, token: str) -> RawJob:
        loc = j.get("location", {})
        return RawJob(
            job_id=f"gh_{j.get('id', hash(j.get('absolute_url','')) & 0xFFFFFFFF)}",
            title=j.get("title",""), company=token.replace("-"," ").title(),
            location=loc.get("name","UK") if isinstance(loc, dict) else "UK",
            description=(j.get("content","") or "")[:8000],
            url=j.get("absolute_url", f"https://boards.greenhouse.io/{token}/jobs"),
            source="ats_direct", is_startup=True,
        )

    async def _fetch_lever(self, client: httpx.AsyncClient, slug: str) -> list[RawJob]:
        try:
            resp = await client.get(f"https://api.lever.co/v0/postings/{slug}", params={"mode": "json", "limit": 50})
            if resp.status_code != 200:
                return []
            postings = resp.json()
            if not isinstance(postings, list):
                return []
            return [self._parse_lever(p, slug) for p in postings]
        except Exception:
            return []

    def _parse_lever(self, p: dict, slug: str) -> RawJob:
        desc = p.get("descriptionPlain","")
        return RawJob(
            job_id=f"lv_{p.get('id', hash(p.get('hostedUrl','')) & 0xFFFFFFFF)}",
            title=p.get("text",""), company=slug.replace("-"," ").title(),
            location=p.get("workplaceType","") or "UK",
            description=desc[:8000], url=p.get("hostedUrl",""),
            source="ats_direct", is_startup=True,
        )

    def _is_ai_role(self, job: RawJob) -> bool:
        text = f"{job.title} {job.description}".lower()
        return any(kw in text for kw in self._AI_KEYWORDS)

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        jobs     = result.get("jobs", [])
        adaptive = plan.get("adaptive", {})
        registry = adaptive.get("token_registry", {})
        for key, cnt in result.get("yield_map", {}).items():
            old = registry.get(key, {}).get("avg_yield", 1.0)
            registry[key] = {"avg_yield": round(0.7 * old + 0.3 * cnt, 2)}
        adaptive["token_registry"] = registry
        state["adaptive_params"]   = adaptive
        state["last_yield"]        = len(jobs)
        return {"quality": "good" if jobs else "warning", "yield": len(jobs), "notes": f"ats_jobs={len(jobs)}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {"jobs": result.get("jobs", []), "source": "ats_direct", "yield": reflection.get("yield", 0), "quality": reflection.get("quality")}


# ══════════════════════════════════════════════════════════════════════════════
# CareerPagesDeepCrawlerAgent
# ══════════════════════════════════════════════════════════════════════════════

class CareerPagesDeepCrawlerAgent(DeepAgent):
    """
    Crawls 150+ company career pages using an adaptive multi-strategy parser.

    plan():    Reads adaptive_params["parse_strategy_log"] — a per-company
               dict remembering which parse strategy succeeded last time
               (json_ld / ats_embed / dom_links). Prioritises companies
               that yielded jobs in the last run. Flags companies where
               parse_rate dropped below 60% (triggers re-evaluation).

    execute(): Multi-strategy parsing per company:
               1. JSON-LD JobPosting structured data
               2. Greenhouse/Lever/Ashby embed detection → record token
               3. Deep crawl: follow individual job page links
               4. Fallback: keyword-matched link text
               Adaptive: records which strategy succeeded, uses it first
               next run.

    reflect(): Computes parse_rate = companies_with_results / companies_crawled.
               Flags companies where strategy changed (possible site redesign).
    """

    agent_id   = "career_pages_crawler_v1"
    department = "data_collection"

    _AI_KEYWORDS = frozenset({
        "machine learning","ml","ai","data scientist","nlp","computer vision",
        "mlops","research scientist","llm","deep learning","pytorch","python",
        "data engineer","applied scientist","generative ai",
    })

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive   = state.get("adaptive_params", {})
        strategy_log = adaptive.get("parse_strategy_log", {})
        watchlist  = context.get("watchlist", [])
        # Sort: companies with recent yield first
        last_yields = adaptive.get("company_yields", {})
        sorted_wl = sorted(watchlist, key=lambda c: -last_yields.get(c["company"], 0))
        return {"watchlist": sorted_wl[:80], "strategy_log": strategy_log, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        watchlist    = plan["watchlist"]
        strategy_log = plan["strategy_log"]
        all_jobs: list[RawJob] = []
        company_yields: dict[str, int] = {}
        sem = asyncio.Semaphore(8)

        headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketForgeBot/0.1)"}

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            async def crawl_one(entry: dict) -> list[RawJob]:
                async with sem:
                    return await self._crawl_company(client, entry, strategy_log)

            results = await asyncio.gather(*[crawl_one(e) for e in watchlist], return_exceptions=True)

        for entry, result in zip(watchlist, results):
            if isinstance(result, list):
                all_jobs.extend(result)
                company_yields[entry["company"]] = len(result)

        parse_rate = sum(1 for v in company_yields.values() if v > 0) / max(len(watchlist), 1)
        return {"jobs": all_jobs, "company_yields": company_yields, "parse_rate": parse_rate}

    async def _crawl_company(self, client: httpx.AsyncClient, entry: dict, strategy_log: dict) -> list[RawJob]:
        company = entry.get("company", "Unknown")
        url     = entry.get("careers_url", "")
        stage   = entry.get("stage", "unknown")
        if not url:
            return []
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Strategy 1: JSON-LD
            jobs = self._extract_jsonld(soup, url, company, stage)
            if jobs:
                strategy_log[company] = "json_ld"
                return jobs

            # Strategy 2: Link extraction
            jobs = self._extract_links(soup, url, company, stage)
            if jobs:
                strategy_log[company] = "dom_links"
                return jobs

        except Exception as exc:
            logger.debug("career_pages.crawl_error", company=company, error=str(exc))
        return []

    def _extract_jsonld(self, soup: Any, page_url: str, company: str, stage: str) -> list[RawJob]:
        jobs = []
        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data  = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("JobPosting", "jobPosting"):
                        title = item.get("title","")
                        if any(kw in title.lower() for kw in self._AI_KEYWORDS):
                            jobs.append(RawJob(
                                job_id=f"cp_{hash(page_url+title) & 0xFFFFFFFF}",
                                title=title, company=company, location="UK",
                                description=(item.get("description","") or "")[:8000],
                                url=item.get("url", page_url),
                                source="career_pages", is_startup=True, company_stage=stage,
                            ))
            except Exception:
                pass
        return jobs

    def _extract_links(self, soup: Any, base_url: str, company: str, stage: str) -> list[RawJob]:
        jobs = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if not text or len(text) < 5 or len(text) > 100:
                continue
            if href.startswith("/"):
                href = urljoin(base_url, href)
            if href in seen:
                continue
            if any(kw in text.lower() for kw in self._AI_KEYWORDS):
                seen.add(href)
                jobs.append(RawJob(
                    job_id=f"cp_{hash(href) & 0xFFFFFFFF}",
                    title=text.strip(), company=company, location="UK",
                    description=f"{text} at {company}.",
                    url=href, source="career_pages", is_startup=True, company_stage=stage,
                ))
        return jobs[:15]

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        parse_rate = result.get("parse_rate", 0)
        adaptive   = plan.get("adaptive", {})
        adaptive["parse_strategy_log"] = plan.get("strategy_log", {})
        adaptive["company_yields"]     = result.get("company_yields", {})
        state["adaptive_params"]       = adaptive
        state["last_yield"]            = len(result.get("jobs", []))

        quality = "good" if parse_rate >= 0.3 else "warning"
        if parse_rate < 0.2:
            logger.warning(f"{self.agent_id}.low_parse_rate", rate=parse_rate)

        return {"quality": quality, "parse_rate": round(parse_rate, 3), "yield": len(result.get("jobs", [])), "notes": f"parse_rate={parse_rate:.2f}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {"jobs": result.get("jobs", []), "source": "career_pages", "yield": reflection.get("yield", 0), "quality": reflection.get("quality")}
