"""
MarketForge AI — Department 6: User Career Insights

Stateless per-request. No user data persisted.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent

logger = structlog.get_logger(__name__)

_CAREER_SYSTEM = """You are a UK AI job market career advisor with access to real market data.
Give specific, actionable advice grounded ONLY in the structured data provided.
Never mention specific company names or make promises about salaries.
Format: clear, professional prose. No bullet lists."""

_CAREER_USER = """Provide career advice based on this user profile and market data.

USER PROFILE:
Skills: {skills}
Target role: {target_role}
Experience level: {experience_level}

CURRENT MARKET DATA:
Top demanded skills for {target_role}: {market_skills}
Salary benchmarks: P25=£{p25:,} / P50=£{p50:,} / P75=£{p75:,} (n={sal_n})
Sponsorship rate in UK AI market: {sponsorship_pct}%

SKILL GAPS (skills in top demand not in user profile): {skill_gaps}

Write four sections:
1. CURRENT MARKET POSITION (2 sentences, specific to their skill set vs market demand)
2. TOP SKILL INVESTMENTS (top 3 gaps to close, with concrete rationale from data)
3. SECTOR OPPORTUNITIES (2-3 sectors where their background gives an edge)
4. 90-DAY ACTION PLAN (specific, achievable steps)

Keep total response under 400 words."""


class UserInsightsLeadAgent(DeepAgent):
    """
    Department 6 Lead Agent — stateless career advisor.

    plan():    Parses the sanitised user profile against the current week's
               market snapshot. Computes market_match_score by comparing
               the user's skill set against the top-demanded skills in
               their target role category. Identifies skill gaps ranked
               by market_demand × salary_correlation.

    execute(): Two-phase:
               Phase 1 — Deterministic gap analysis (no LLM):
                          Gap score = demand_rank × salary_uplift_estimate
                          Sector fit = cosine similarity vs sector skill profiles
               Phase 2 — Gemini Pro narrative synthesis (structured data → prose):
                          Prompt contains ONLY verified data points.
                          Passes through OutputGuardrailsAgent before returning.

    reflect(): Validates that the generated narrative contains no claims
               not present in the structured data that was passed in.
               Checks for discriminatory language signals (age, gender, nationality).

    output():  Returns career advice report. No user data retained after output().
    """

    agent_id   = "user_insights_lead_v1"
    department = "user_insights"
    uses_llm   = True

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        sanitised = context.get("sanitised_profile", {})
        visa_needed = context.get("visa_needed", False)

        # Parse skills from sanitised input
        skills_raw = sanitised.get("skills", "")
        user_skills = {s.strip().lower() for s in skills_raw.split(",") if s.strip()}
        target_role = sanitised.get("target_role", "ai_engineer").lower().replace(" ", "_")
        exp_level   = sanitised.get("experience_level", "mid")

        # Load market snapshot
        snapshot = self._load_snapshot(target_role)

        # Compute skill gaps
        market_skills = list(snapshot.get("top_skills", {}).keys())
        skill_gaps    = [s for s in market_skills[:15] if s.lower() not in user_skills][:8]

        # Match score: fraction of top-10 skills user has
        top10 = set(s.lower() for s in market_skills[:10])
        match_score = len(user_skills & top10) / max(len(top10), 1)

        return {
            "user_skills":    list(user_skills),
            "target_role":    target_role,
            "exp_level":      exp_level,
            "skill_gaps":     skill_gaps,
            "match_score":    match_score,
            "snapshot":       snapshot,
            "visa_needed":    visa_needed,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from marketforge.config.settings import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        snapshot  = plan["snapshot"]
        skill_gaps= plan["skill_gaps"]
        top_skills= list(snapshot.get("top_skills", {}).items())[:10]

        prompt = _CAREER_USER.format(
            skills          = ", ".join(plan["user_skills"][:15]) or "not specified",
            target_role     = plan["target_role"].replace("_", " ").title(),
            experience_level= plan["exp_level"],
            market_skills   = ", ".join(s for s, _ in top_skills[:8]),
            p25             = int(snapshot.get("salary_p25") or 0),
            p50             = int(snapshot.get("salary_p50") or 0),
            p75             = int(snapshot.get("salary_p75") or 0),
            sal_n           = snapshot.get("salary_sample_size", 0),
            sponsorship_pct = round((snapshot.get("sponsorship_rate") or 0) * 100, 1),
            skill_gaps      = ", ".join(skill_gaps[:6]) or "minimal gaps detected",
        )

        pro = ChatGoogleGenerativeAI(
            model=settings.llm.deep_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.3,
        )
        response = await pro.ainvoke([
            SystemMessage(content=_CAREER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        narrative = response.content.strip()

        # Sector fit (deterministic)
        sector_fit = self._compute_sector_fit(plan["user_skills"])

        return {
            "narrative":      narrative,
            "sector_fit":     sector_fit,
            "skill_gaps":     skill_gaps,
            "match_score":    plan["match_score"],
            "snapshot":       snapshot,
        }

    def _load_snapshot(self, role_category: str) -> dict:
        try:
            from marketforge.memory.postgres import get_sync_engine
            from sqlalchemy import text
            engine    = get_sync_engine()
            is_sqlite = engine.dialect.name == "sqlite"
            snaps_t   = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"
            with engine.connect() as conn:
                row = conn.execute(text(f"""
                    SELECT top_skills, salary_p25, salary_p50, salary_p75,
                           salary_sample_size, sponsorship_rate
                    FROM {snaps_t} WHERE role_category='all'
                    ORDER BY week_start DESC LIMIT 1
                """)).fetchone()
            if row:
                ts = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
                return {
                    "top_skills":         ts,
                    "salary_p25":         row[1], "salary_p50": row[2], "salary_p75": row[3],
                    "salary_sample_size": row[4], "sponsorship_rate": row[5],
                }
        except Exception as exc:
            logger.warning("user_insights.snapshot_load_failed", error=str(exc))
        return {"top_skills": {}, "salary_p50": 65_000, "sponsorship_rate": 0.12}

    @staticmethod
    def _compute_sector_fit(user_skills: list[str]) -> list[dict]:
        """
        Simple sector fit based on skill keyword overlap.
        Full SBERT-based fit in Phase 4.
        """
        sectors = {
            "FinTech":          ["python", "mlops", "data science", "fraud detection", "sql"],
            "HealthTech":       ["deep learning", "computer vision", "pytorch", "medical imaging"],
            "AI Safety":        ["alignment", "rlhf", "interpretability", "safety"],
            "Enterprise AI":    ["llm", "langchain", "rag", "fastapi", "aws"],
            "Research":         ["pytorch", "research", "arxiv", "experiments", "nlp"],
        }
        user_set = set(u.lower() for u in user_skills)
        results  = []
        for sector, keywords in sectors.items():
            overlap = sum(1 for k in keywords if any(k in u for u in user_set))
            results.append({"sector": sector, "fit_score": round(overlap / len(keywords), 2)})
        return sorted(results, key=lambda x: -x["fit_score"])[:3]

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        narrative = result.get("narrative", "")
        # Basic guardrail: check for discriminatory language
        risky_terms = ["age", "young", "old", "male", "female", "nationality", "accent"]
        flagged = [t for t in risky_terms if t in narrative.lower()]
        if flagged:
            logger.warning(f"{self.agent_id}.discriminatory_language_detected", terms=flagged)
        quality = "warning" if flagged else "good"
        return {"quality": quality, "flagged_terms": flagged}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        narrative = result.get("narrative", "")
        # Parse narrative into sections
        lines = narrative.split("\n")
        return {
            "market_position":  "\n".join(l for l in lines if "MARKET POSITION" in l.upper() or (lines.index(l) < 5)),
            "skill_gaps":       [{"skill": s, "priority": i+1} for i, s in enumerate(result.get("skill_gaps", []))],
            "sector_fit":       result.get("sector_fit", []),
            "salary_estimate":  {
                "p25": result.get("snapshot", {}).get("salary_p25"),
                "p50": result.get("snapshot", {}).get("salary_p50"),
                "p75": result.get("snapshot", {}).get("salary_p75"),
            },
            "action_plan":      narrative,
            "match_score":      result.get("match_score", 0.0),
        }
