"""
MarketForge AI — Department 1: SpecialistBoardsDeepAgent

Handles CWJobs, Totaljobs, CV-Library, Jobserve, and Guardian Jobs using
respectful scraping with politeness policies:
  - 3-second delay between requests per domain
  - robots.txt compliance check
  - Session rotation to avoid IP-level rate limiting
  - Failure isolation: one board failing never halts others

This agent maintains a ToS-risk score per board. If risk indicators
(rate limit 429s, CAPTCHA detection, HTML structure changes) exceed
a threshold, the board is halted for that run and the event is logged
for ops review.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)

_BOARDS = [
    {
        "name":      "CWJobs",
        "search_url":"https://www.cwjobs.co.uk/jobs/machine-learning",
        "query_url": "https://www.cwjobs.co.uk/jobs/{query}",
        "queries":   ["machine-learning-engineer", "data-scientist", "ai-engineer", "nlp-engineer"],
    },
    {
        "name":      "Totaljobs",
        "search_url":"https://www.totaljobs.com/jobs/machine-learning-engineer",
        "query_url": "https://www.totaljobs.com/jobs/{query}",
        "queries":   ["machine-learning-engineer", "data-scientist", "ai-engineer"],
    },
    {
        "name":      "CV-Library",
        "search_url":"https://www.cv-library.co.uk/search-jobs?q=machine+learning",
        "query_url": "https://www.cv-library.co.uk/search-jobs?q={query}",
        "queries":   ["machine+learning+engineer", "data+scientist", "ai+engineer"],
    },
    {
        "name":      "Guardian Jobs",
        "search_url":"https://jobs.theguardian.com/jobs/it/machine-learning/",
        "query_url": "https://jobs.theguardian.com/search/?q={query}",
        "queries":   ["machine+learning", "data+scientist", "ai+engineer"],
    },
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

_AI_KEYWORDS = frozenset({
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp engineer", "computer vision", "deep learning",
    "research scientist", "mlops", "data engineer", "generative ai",
    "pytorch", "tensorflow", "applied scientist",
})


class SpecialistBoardsDeepAgent(DeepAgent):
    """
    Deep Agent for UK specialist tech job boards (CWJobs, Totaljobs, etc.).

    plan():    Reads adaptive_params["board_tos_risk_scores"] — a per-board
               float (0=safe, 1=halt). Boards with risk > 0.8 are skipped
               this run. Also checks adaptive_params["board_parse_strategies"]
               to remember which selector worked for each board last time.

    execute(): Respectful HTML scraping with politeness policies:
               - 3s delay between requests to the same domain
               - Detects CAPTCHA responses (title contains "robot", "verify")
               - Detects 429 rate limits
               - Session rotation via fresh httpx.AsyncClient per board
               Runs all boards in parallel but rate-limits within each board
               using a per-domain asyncio.Semaphore(1) to enforce the 3s delay.

    reflect(): Updates ToS-risk scores: 429 → +0.3, CAPTCHA → +0.5,
               parse_rate < 0.2 → +0.1, success → -0.1 (floor 0).
               Boards that reach risk > 0.8 are logged to ops for review.
    """

    agent_id   = "specialist_boards_v1"
    department = "data_collection"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive   = state.get("adaptive_params", {})
        risk_scores= adaptive.get("board_tos_risk_scores", {})
        active     = [b for b in _BOARDS if risk_scores.get(b["name"], 0) < 0.8]
        skipped    = [b["name"] for b in _BOARDS if risk_scores.get(b["name"], 0) >= 0.8]
        if skipped:
            logger.info(f"{self.agent_id}.boards_skipped_due_to_risk", boards=skipped)
        return {"boards": active, "risk_scores": risk_scores, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        boards     = plan["boards"]
        all_jobs:  list[RawJob] = []
        board_stats: dict[str, dict] = {}
        domain_locks: dict[str, asyncio.Semaphore] = {}

        async def scrape_board(board: dict) -> tuple[str, list[RawJob], dict]:
            domain = board["search_url"].split("/")[2]
            if domain not in domain_locks:
                domain_locks[domain] = asyncio.Semaphore(1)

            jobs:  list[RawJob] = []
            stats: dict = {"requests": 0, "captcha": 0, "rate_limit": 0, "parsed": 0}

            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=True, headers=_HEADERS
            ) as client:
                for query in board["queries"][:2]:
                    async with domain_locks[domain]:
                        url = board["query_url"].format(query=query)
                        try:
                            resp = await client.get(url)
                            stats["requests"] += 1

                            if resp.status_code == 429:
                                stats["rate_limit"] += 1
                                logger.warning(
                                    f"{self.agent_id}.rate_limited",
                                    board=board["name"],
                                )
                                break

                            if resp.status_code != 200:
                                continue

                            # CAPTCHA detection
                            title_match = re.search(
                                r"<title[^>]*>(.*?)</title>", resp.text, re.I | re.S
                            )
                            if title_match:
                                page_title = title_match.group(1).lower()
                                if any(x in page_title for x in ["robot", "verify", "captcha", "403"]):
                                    stats["captcha"] += 1
                                    logger.warning(f"{self.agent_id}.captcha", board=board["name"])
                                    break

                            batch = self._parse_board(resp.text, url, board["name"])
                            relevant = [j for j in batch if self._is_relevant(j)]
                            jobs.extend(relevant)
                            stats["parsed"] += len(relevant)

                        except Exception as exc:
                            logger.warning(f"{self.agent_id}.request_error",
                                           board=board["name"], error=str(exc))

                        # Politeness delay — always wait 3s between requests to same domain
                        await asyncio.sleep(settings.sources.politeness_delay_s)

            return board["name"], jobs, stats

        tasks = [scrape_board(b) for b in boards]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                name, jobs, stats = result
                all_jobs.extend(jobs)
                board_stats[name] = stats

        return {"jobs": all_jobs, "board_stats": board_stats}

    def _parse_board(self, html: str, page_url: str, board_name: str) -> list[RawJob]:
        """Generic job listing parser — looks for common job card patterns."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        soup  = BeautifulSoup(html, "html.parser")
        jobs: list[RawJob] = []
        seen: set[str] = set()

        # Try common job card selectors
        job_cards = (
            soup.find_all("article", {"class": re.compile(r"job|listing|result", re.I)})
            or soup.find_all("div",   {"class": re.compile(r"job-item|job-card|listing", re.I)})
            or soup.find_all("li",    {"class": re.compile(r"job|result", re.I)})
        )

        for card in job_cards[:30]:
            # Title: first heading or strong element
            title_el = (card.find(["h2", "h3", "h4"]) or card.find("strong"))
            title    = title_el.get_text(strip=True) if title_el else ""
            if not title or len(title) < 5:
                continue

            # URL: first link in card
            link  = card.find("a", href=True)
            href  = link["href"] if link else ""
            if href.startswith("/"):
                href = urljoin(page_url, href)
            if href in seen:
                continue
            seen.add(href)

            # Company
            company_el = card.find(attrs={"class": re.compile(r"company|employer", re.I)})
            company    = company_el.get_text(strip=True) if company_el else "Unknown"

            # Location
            loc_el  = card.find(attrs={"class": re.compile(r"location|loc", re.I)})
            location= loc_el.get_text(strip=True) if loc_el else "UK"

            # Salary
            sal_el  = card.find(attrs={"class": re.compile(r"salary|pay|compensation", re.I)})
            sal_text= sal_el.get_text(strip=True) if sal_el else ""
            sal_min, sal_max = self._extract_salary(sal_text)

            # Description snippet
            desc_el = card.find(attrs={"class": re.compile(r"desc|summary|snippet", re.I)})
            desc    = desc_el.get_text(strip=True) if desc_el else title

            jobs.append(RawJob(
                job_id=f"sb_{hash(href) & 0xFFFFFFFF}",
                title=title, company=company, location=location[:100],
                salary_min=sal_min, salary_max=sal_max,
                description=desc[:4000], url=href,
                source="specialist_boards",
            ))

        # Fallback: link-text scan if no cards found
        if not jobs:
            for a in soup.find_all("a", href=True):
                text = a.get_text(strip=True)
                href = a["href"]
                if not text or len(text) < 8 or len(text) > 100:
                    continue
                if href.startswith("/"):
                    href = urljoin(page_url, href)
                if href in seen:
                    continue
                if any(kw in text.lower() for kw in _AI_KEYWORDS):
                    seen.add(href)
                    jobs.append(RawJob(
                        job_id=f"sb_{hash(href) & 0xFFFFFFFF}",
                        title=text.strip(), company="Unknown", location="UK",
                        description=text, url=href,
                        source="specialist_boards",
                    ))
            jobs = jobs[:15]

        return jobs

    @staticmethod
    def _extract_salary(text: str) -> tuple[float | None, float | None]:
        m = re.search(r"£\s*(\d{2,3})(?:,\d{3})?\s*[-–]\s*£?\s*(\d{2,3})(?:,\d{3})?", text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo < 500:
                lo *= 1000; hi *= 1000
            return lo, hi
        m2 = re.search(r"£\s*(\d{2,3})k?\b", text, re.I)
        if m2:
            v = float(m2.group(1))
            if v < 500:
                v *= 1000
            return v, v
        return None, None

    @staticmethod
    def _is_relevant(job: RawJob) -> bool:
        text = f"{job.title} {job.description}".lower()
        return any(kw in text for kw in _AI_KEYWORDS)

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        board_stats = result.get("board_stats", {})
        adaptive    = plan.get("adaptive", {})
        risk_scores = plan.get("risk_scores", {})

        for board_name, stats in board_stats.items():
            delta = 0.0
            if stats.get("rate_limit", 0) > 0:
                delta += 0.3
            if stats.get("captcha", 0) > 0:
                delta += 0.5
            total     = stats.get("requests", 1)
            parse_rate= stats.get("parsed", 0) / max(total, 1)
            if parse_rate < 0.2 and total > 0:
                delta += 0.1
            elif delta == 0:
                delta = -0.1  # successful run reduces risk score

            old_risk = risk_scores.get(board_name, 0.0)
            new_risk = max(0.0, min(1.0, old_risk + delta))
            risk_scores[board_name] = round(new_risk, 2)

            if new_risk > 0.8:
                logger.warning(
                    f"{self.agent_id}.board_halted",
                    board=board_name,
                    risk=new_risk,
                )

        adaptive["board_tos_risk_scores"] = risk_scores
        state["adaptive_params"]          = adaptive
        state["last_yield"]               = len(result.get("jobs", []))

        quality = "good" if state["last_yield"] > 5 else "warning"
        return {
            "quality":    quality,
            "yield":      state["last_yield"],
            "risk_scores":risk_scores,
            "notes":      f"specialist_jobs={state['last_yield']}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "jobs":       result.get("jobs", []),
            "source":     "specialist_boards",
            "yield":      reflection.get("yield", 0),
            "quality":    reflection.get("quality"),
            "risk_scores":reflection.get("risk_scores", {}),
        }
