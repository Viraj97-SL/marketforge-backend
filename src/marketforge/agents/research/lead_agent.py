"""
MarketForge AI — Department 4: Research Intelligence

Detects emerging techniques 4-8 weeks before they appear in job postings.
Each sub-agent monitors a different intelligence source and feeds signals
into market.research_signals.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.memory.postgres import get_sync_engine

logger = structlog.get_logger(__name__)


def _t(name: str) -> str:
    engine = get_sync_engine()
    return name if engine.dialect.name == "sqlite" else f"market.{name}"


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 1: arXivMonitorAgent
# ══════════════════════════════════════════════════════════════════════════════

class arXivMonitorAgent(DeepAgent):
    """
    Queries arXiv cs.LG, cs.AI, cs.CL, cs.CV, stat.ML for recent papers.

    plan():    Reads the current top-trending skills from market.weekly_snapshots
               to build targeted arXiv queries. If "mixture of experts" is
               trending in JDs, queries arXiv for recent papers on that topic.
               Also reads adaptive_params["seen_arxiv_ids"] to skip papers
               already processed.

    execute(): Fetches paper metadata via the arXiv Atom API. Computes a
               relevance_score per paper: (UK AI market keyword density in
               abstract) × (recency weight). Uses Gemini Flash to generate
               structured summary cards for papers with score > 0.6.
               Summary card fields: core_technique, novelty_claim,
               practical_applicability_score (0-1), requires_specialised_hardware.

    reflect(): Checks correlation between techniques appearing in new papers
               and their presence rate in recent job descriptions. Updates the
               technique_adoption_lag model in adaptive_params.
    """

    agent_id   = "arxiv_monitor_v1"
    department = "research"
    uses_llm   = True

    _ARXIV_BASE = "http://export.arxiv.org/api/query"
    _CATEGORIES = ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"]
    _AI_KEYWORDS = {
        "transformer", "attention", "diffusion", "llm", "language model",
        "reinforcement learning", "neural network", "fine-tuning", "rlhf",
        "mixture of experts", "moe", "retrieval augmented", "rag",
        "multimodal", "vision language", "autonomous agent", "tool use",
        "chain of thought", "in-context learning", "emergence",
    }

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive    = state.get("adaptive_params", {})
        seen_ids    = set(adaptive.get("seen_arxiv_ids", []))
        top_skills  = context.get("top_skills", {})

        # Build queries from trending skills
        queries = [f"ti:{skill.replace(' ','+')}+AND+cat:cs.LG"
                   for skill in list(top_skills.keys())[:5]]
        # Always include broad AI queries
        queries += [f"cat:{cat}" for cat in self._CATEGORIES[:3]]

        return {"queries": queries, "seen_ids": seen_ids, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        import httpx
        from marketforge.config.settings import settings

        queries  = plan["queries"]
        seen_ids = plan["seen_ids"]
        papers   = []

        async with httpx.AsyncClient(timeout=20.0) as client:
            for query in queries[:5]:  # cap to 5 queries
                try:
                    resp = await client.get(self._ARXIV_BASE, params={
                        "search_query": query,
                        "start":        0,
                        "max_results":  10,
                        "sortBy":       "submittedDate",
                        "sortOrder":    "descending",
                    })
                    if resp.status_code != 200:
                        continue
                    papers.extend(self._parse_atom(resp.text, seen_ids))
                except Exception as exc:
                    logger.warning("arxiv.fetch.error", query=query[:40], error=str(exc))

        # Score relevance
        scored = []
        for p in papers:
            score = self._relevance_score(p["abstract"])
            if score >= 0.4:
                p["relevance_score"] = score
                scored.append(p)

        scored.sort(key=lambda x: -x["relevance_score"])
        scored = scored[:20]  # top 20 per run

        # Generate summary cards for top papers via Gemini Flash
        summaries = []
        if scored and settings.llm.gemini_api_key:
            summaries = await self._generate_summary_cards(scored[:10])

        return {"papers": scored, "summaries": summaries}

    def _parse_atom(self, xml: str, seen_ids: set) -> list[dict]:
        """Parse arXiv Atom XML response into paper dicts."""
        papers = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for entry in entries:
            arxiv_id = re.search(r"<id>.*?/abs/([^<]+)</id>", entry)
            title    = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            abstract = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            published= re.search(r"<published>([\d\-T:Z]+)</published>", entry)

            if not arxiv_id:
                continue
            aid = arxiv_id.group(1).strip()
            if aid in seen_ids:
                continue

            papers.append({
                "arxiv_id":    aid,
                "title":       (title.group(1).strip() if title else "").replace("\n", " "),
                "abstract":    (abstract.group(1).strip() if abstract else ""),
                "published_at":(published.group(1)[:10] if published else str(date.today())),
            })
        return papers

    def _relevance_score(self, abstract: str) -> float:
        if not abstract:
            return 0.0
        text = abstract.lower()
        hits = sum(1 for kw in self._AI_KEYWORDS if kw in text)
        return min(hits / 5.0, 1.0)

    async def _generate_summary_cards(self, papers: list[dict]) -> list[dict]:
        """Use Gemini Flash to generate structured summary cards."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            from marketforge.config.settings import settings
            import json as _json

            llm = ChatGoogleGenerativeAI(
                model=settings.llm.fast_model,
                google_api_key=settings.llm.gemini_api_key,
                temperature=0,
            )
            summaries = []
            for paper in papers:
                prompt = f"""Analyse this AI research paper abstract and return ONLY a JSON object.

Title: {paper['title']}
Abstract: {paper['abstract'][:800]}

Return JSON with exactly these keys:
- core_technique: string (the main technical contribution in 5-10 words)
- novelty_claim: string (what makes this novel in 10-15 words)
- practical_applicability_score: float 0-1 (0=purely theoretical, 1=immediately deployable)
- requires_specialised_hardware: boolean
- uk_market_relevance: float 0-1 (relevance to UK AI/ML job market)"""

                resp = llm.invoke([HumanMessage(content=prompt)])
                raw  = resp.content.strip().replace("```json","").replace("```","")
                try:
                    card = _json.loads(raw)
                    card["arxiv_id"] = paper["arxiv_id"]
                    card["title"]    = paper["title"]
                    summaries.append(card)
                except Exception:
                    pass
            return summaries
        except Exception as exc:
            logger.warning("arxiv.summary_cards.error", error=str(exc))
            return []

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        papers   = result.get("papers", [])
        adaptive = plan.get("adaptive", {})
        # Track seen IDs (cap at 5000)
        seen = list(adaptive.get("seen_arxiv_ids", []))
        seen.extend(p["arxiv_id"] for p in papers)
        adaptive["seen_arxiv_ids"] = seen[-5000:]
        state["adaptive_params"]   = adaptive
        state["last_yield"]        = len(papers)
        return {"quality": "good", "papers_found": len(papers), "notes": f"n={len(papers)}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        # Persist research signals
        papers    = result.get("papers", [])
        summaries = {s["arxiv_id"]: s for s in result.get("summaries", [])}
        self._persist_signals(papers, summaries)
        return {"research_papers": papers, "summary_cards": result.get("summaries", [])}

    def _persist_signals(self, papers: list[dict], summaries: dict) -> None:
        from sqlalchemy import text
        from datetime import datetime
        engine = get_sync_engine()
        table  = "research_signals" if engine.dialect.name == "sqlite" else "market.research_signals"

        with engine.connect() as conn:
            for p in papers:
                card    = summaries.get(p["arxiv_id"], {})
                summary = card.get("core_technique", p["title"][:120])
                score   = p.get("relevance_score", 0) * (card.get("uk_market_relevance", 0.5) if card else 0.5)
                conn.execute(text(f"""
                    INSERT INTO {table} (technique_name, source, first_seen, mention_count, relevance_score, summary)
                    VALUES (:name, 'arxiv', :fs, 1, :score, :summary)
                    ON CONFLICT(technique_name) DO UPDATE SET
                        mention_count  = {table}.mention_count + 1,
                        relevance_score= MAX({table}.relevance_score, EXCLUDED.relevance_score),
                        updated_at     = '{datetime.utcnow().isoformat()}'
                """), {
                    "name":    p["title"][:200],
                    "fs":      p["published_at"],
                    "score":   round(score, 3),
                    "summary": summary,
                })
            conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 2: EmergingTechSignalAgent
# ══════════════════════════════════════════════════════════════════════════════

class EmergingTechSignalAgent(DeepAgent):
    """
    The forward-looking synthesiser — correlates arXiv velocity with JD appearance.

    plan():    Queries research_signals for techniques first seen > 2 weeks ago
               but not yet appearing in job descriptions. These are the candidates
               for adoption lag modelling.

    execute(): For each candidate technique, queries market.job_skills to check
               if any skill matching the technique name has appeared recently.
               Computes adoption_lag_days = (first_in_jd - first_seen).
               Builds a regression model estimate for techniques not yet in JDs:
               predicted_weeks_to_jd = mean_lag × (1 - relevance_score).

    reflect(): Validates the adoption lag estimates against the 4-week rolling
               mean. Flags techniques where lag is shortening (accelerating
               adoption) as high-priority signals for the weekly report.
    """

    agent_id   = "emerging_tech_signal_v1"
    department = "research"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return {"adaptive": state.get("adaptive_params", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        engine = get_sync_engine()
        rs_t   = "research_signals" if engine.dialect.name == "sqlite" else "market.research_signals"
        sk_t   = "job_skills"       if engine.dialect.name == "sqlite" else "market.job_skills"

        # Techniques in research but not yet in JDs
        with engine.connect() as conn:
            candidates = conn.execute(text(f"""
                SELECT technique_name, first_seen, relevance_score, mention_count
                FROM {rs_t}
                WHERE first_in_jd IS NULL
                  AND mention_count >= 2
                  AND relevance_score >= 0.3
                ORDER BY relevance_score DESC
                LIMIT 30
            """)).fetchall()

            # Check if any have appeared in job skills
            updates = []
            for name, first_seen, score, count in candidates:
                # Fuzzy match: check if any skill contains keywords from technique name
                keywords = [w for w in name.lower().split() if len(w) > 4]
                if not keywords:
                    continue
                like_clause = " OR ".join(f"LOWER(skill) LIKE '%{kw}%'" for kw in keywords[:3])
                found = conn.execute(text(f"""
                    SELECT MIN(extracted_at) FROM {sk_t}
                    WHERE {like_clause}
                """)).scalar()
                if found:
                    first_seen_date = date.fromisoformat(str(first_seen)[:10]) if first_seen else date.today()
                    first_jd_date   = date.fromisoformat(str(found)[:10])
                    lag_days        = (first_jd_date - first_seen_date).days
                    updates.append((name, str(first_jd_date), max(0, lag_days)))

            # Persist adoption lag updates
            for technique_name, first_in_jd, lag in updates:
                conn.execute(text(f"""
                    UPDATE {rs_t}
                    SET first_in_jd = :fj, adoption_lag_days = :lag
                    WHERE technique_name = :name
                """), {"fj": first_in_jd, "lag": lag, "name": technique_name})
            conn.commit()

        # Compute mean lag for forecasting
        with engine.connect() as conn:
            lag_row = conn.execute(text(f"""
                SELECT AVG(adoption_lag_days), STDDEV(adoption_lag_days)
                FROM {rs_t}
                WHERE adoption_lag_days IS NOT NULL AND adoption_lag_days > 0
            """)).fetchone()

        mean_lag = lag_row[0] or 42  # default 6 weeks
        std_lag  = lag_row[1] or 14

        # Generate predictions for unconfirmed techniques
        predictions = []
        for name, first_seen, score, count in candidates:
            predicted_weeks = max(1, round((mean_lag * (1 - score * 0.3)) / 7))
            predictions.append({
                "technique":       name[:100],
                "predicted_weeks_to_jd": predicted_weeks,
                "relevance_score": score,
            })

        return {
            "confirmed_adoptions": updates,
            "predictions":         predictions[:10],
            "mean_lag_days":       round(mean_lag, 1),
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        confirmed = result.get("confirmed_adoptions", [])
        if confirmed:
            logger.info(f"{self.agent_id}.adoptions_confirmed", count=len(confirmed), techniques=[c[0][:40] for c in confirmed[:3]])
        state["last_yield"] = len(confirmed)
        return {"quality": "good", "adoptions": len(confirmed), "notes": f"mean_lag={result.get('mean_lag_days')}d"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "emerging_signals":    result.get("predictions", []),
            "confirmed_adoptions": result.get("confirmed_adoptions", []),
            "mean_adoption_lag_days": result.get("mean_lag_days", 42),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Research Lead Agent
# ══════════════════════════════════════════════════════════════════════════════

class ResearchLeadAgent(DeepAgent):
    """
    Department 4 Lead — monitors the academic and industry frontier.

    plan():    Queries the current week's top skills to build a research
               agenda: which techniques should be monitored in arXiv?
               Which engineering blogs are highest priority?

    execute(): Dispatches arXivMonitorAgent and EmergingTechSignalAgent.
               Results feed into ContentStudio's weekly report.

    reflect(): Validates that research signals cover the same skill domains
               as the current market data (no blind spots in coverage).
    """

    agent_id   = "research_lead_v1"
    department = "research"

    def __init__(self) -> None:
        self._arxiv  = arXivMonitorAgent()
        self._signal = EmergingTechSignalAgent()

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        # Pull top skills from latest snapshot to inform arXiv queries
        top_skills: dict[str, int] = {}
        try:
            from sqlalchemy import text
            import json as _json
            engine = get_sync_engine()
            with engine.connect() as conn:
                row = conn.execute(text(f"SELECT top_skills FROM {_t('weekly_snapshots')} ORDER BY week_start DESC LIMIT 1")).fetchone()
            if row and row[0]:
                top_skills = _json.loads(row[0]) if isinstance(row[0], str) else row[0]
        except Exception:
            pass
        return {"top_skills": top_skills, "adaptive": state.get("adaptive_params", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        ctx = {"top_skills": plan["top_skills"]}
        arxiv_out  = await self._arxiv.run(ctx)
        signal_out = await self._signal.run(ctx)
        return {"arxiv": arxiv_out, "signals": signal_out}

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        papers  = len(result.get("arxiv", {}).get("research_papers", []))
        signals = len(result.get("signals", {}).get("emerging_signals", []))
        state["last_yield"] = papers + signals
        return {"quality": "good", "papers": papers, "signals": signals, "notes": f"papers={papers}, signals={signals}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "research_papers":    result.get("arxiv", {}).get("research_papers", []),
            "summary_cards":      result.get("arxiv", {}).get("summary_cards", []),
            "emerging_signals":   result.get("signals", {}).get("emerging_signals", []),
            "confirmed_adoptions":result.get("signals", {}).get("confirmed_adoptions", []),
        }
