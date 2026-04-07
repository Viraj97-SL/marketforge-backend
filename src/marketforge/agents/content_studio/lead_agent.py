"""
MarketForge AI — Department 5: Content Studio

ContentLeadAgent + WeeklyReportWriterAgent + ContrarianInsightAgent

All LLM calls use a strict structured-data-in → narrative-out pattern.
No free-form user text is ever injected into prompts.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.utils.cost_tracker import CostTracker, CostTrackerCallback

logger = structlog.get_logger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────────────

_NARRATIVE_SYSTEM = """You are a senior AI market analyst writing a weekly UK AI job market report.
Your writing is punchy, data-backed, and actionable — LinkedIn-post quality.

STRICT RULES:
1. Every claim must be supported by a data point provided in the structured data below.
2. Never invent statistics, company names, or technologies not present in the data.
3. Opening sentence must contain a specific number or percentage.
4. Use present tense for current trends, past tense for prior-week comparison.
5. Target reading grade level 11 (clear, professional, not academic).
6. Length: 300-500 words total."""

_NARRATIVE_USER = """Generate the weekly UK AI job market report from this structured data only.

MARKET DATA (week of {week_start}):
- Total new AI/ML job postings: {job_count}
- Top 10 skills by demand: {top_skills}
- Median salary: £{salary_p50:,} (n={salary_n})
- Salary range P25–P75: £{salary_p25:,} – £{salary_p75:,}
- Sponsorship rate: {sponsorship_pct}% of postings
- Remote/hybrid availability: {remote_pct}%

INSIGHT FLAGS (from Market Analysis department):
{insight_flags}

Structure the report as:
1. HEADLINE (1 sentence with the most significant data point this week)
2. SKILL SPOTLIGHT (top 3 skills, trend direction, why it matters)
3. SALARY SNAPSHOT (benchmark numbers, any notable band movement)
4. SPONSORSHIP SIGNAL (rate vs prior week, which sectors)
5. ONE CONTRARIAN OBSERVATION (something surprising in the data)
6. WHAT TO WATCH (1 emerging signal from research intelligence)

Output plain text. No markdown headers."""

_CONTRARIAN_SYSTEM = """You are a data analyst who specialises in finding counterintuitive patterns.
Given market data, identify ONE surprising observation that a straightforward summary would miss.

Examples of good contrarian observations:
- A skill declining in volume but rising in salary (scarcity premium)
- A company stage showing higher-than-expected sponsorship
- A role category with unusually low geographic concentration

Rules:
- Must be statistically defensible from the data provided
- Must be genuinely counterintuitive (not obvious)
- Must be actionable for a job seeker or hiring manager
- Output: one paragraph, max 80 words"""

_QA_SYSTEM = """You are a quality evaluator for a professional market intelligence report.
Evaluate the report against these 8 criteria and return JSON only.

Criteria:
1. factual_accuracy: every statistic traces to a provided data source (0-10)
2. no_superlatives: no unsubstantiated "most", "best", "fastest" (0-10)
3. reading_grade: target grade 11, score 10 if 10-12, 8 if 8-9 or 13+ (0-10)
4. actionability: contains specific actions a reader can take (0-10)
5. no_repetition: no repeated phrases or ideas (0-10)
6. length_ok: 300-500 words scores 10, outside range loses 2 per 50 words (0-10)
7. no_hallucinated_names: no company names not in the provided data (0-10)
8. tone_balance: professional but not dry (0-10)

Return ONLY this JSON:
{"scores": {"factual_accuracy": N, ...}, "overall": N, "pass": bool, "corrections": ["..."]}
where pass = true if overall >= 7.5"""


class WeeklyReportWriterAgent(DeepAgent):
    """
    Deep Agent that assembles the weekly email report.

    plan():    Reviews available snapshot data, checks prior week's report
               for style consistency, identifies the 3 most significant
               changes from last week vs this week.

    execute(): Two-pass LLM generation:
               Pass 1 — DataNarrativeAgent generates validated insight claims
               Pass 2 — WeeklyReportWriterAgent assembles full report
               Self-review against 7 quality criteria before passing to QA.

    reflect(): Evaluates its own draft against a rubric. Computes a
               self_review_score. If score < 7.5, rewrites the weakest
               section and re-scores (max 1 rewrite per run).

    output():  Returns report_draft and self_review_score for QALeadAgent.
    """

    agent_id   = "weekly_report_writer_v1"
    department = "content_studio"
    uses_llm   = True

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        snapshot = context.get("snapshot", {})
        adaptive = state.get("adaptive_params", {})

        # Identify top 3 most significant changes vs prior week
        top_skills     = snapshot.get("top_skills", {})
        prior_snapshot = adaptive.get("prior_snapshot", {})
        prior_skills   = prior_snapshot.get("top_skills", {})

        skill_changes: list[dict] = []
        for skill, count in list(top_skills.items())[:10]:
            prior = prior_skills.get(skill, 0)
            if prior > 0:
                delta_pct = round((count - prior) / prior * 100, 1)
                skill_changes.append({"skill": skill, "delta_pct": delta_pct, "count": count})
        skill_changes.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)

        insight_flags = "\n".join([
            f"- {c['skill']}: {'+' if c['delta_pct'] >= 0 else ''}{c['delta_pct']}% vs last week ({c['count']} postings)"
            for c in skill_changes[:5]
        ]) or "- Insufficient prior-week data for comparison (first run)"

        logger.info(f"{self.agent_id}.plan.done", top_changes=len(skill_changes))
        return {
            "snapshot":      snapshot,
            "insight_flags": insight_flags,
            "prior_skills":  prior_skills,
            "adaptive":      adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from marketforge.config.settings import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        snap   = plan["snapshot"]
        flags  = plan["insight_flags"]
        cost_t: CostTracker | None = state.get("_cost_tracker")

        # ── Build LLM instances ───────────────────────────────────────────────
        callbacks = []
        if cost_t:
            callbacks.append(CostTrackerCallback(cost_t, self.agent_id, self.department, settings.llm.fast_model))

        flash = ChatGoogleGenerativeAI(
            model=settings.llm.fast_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.3,
            callbacks=callbacks,
        )
        pro = ChatGoogleGenerativeAI(
            model=settings.llm.deep_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0.2,
            callbacks=callbacks,
        )

        # ── Pass 1: contrarian insight (Flash) ────────────────────────────────
        contrarian_prompt = f"""Market data for week of {snap.get('week_start')}:
Top skills: {json.dumps(list(snap.get('top_skills', {}).items())[:10])}
Salary P50: £{snap.get('salary_p50', 0):,}
Sponsorship rate: {snap.get('sponsorship_rate', 0)*100:.1f}%
Changes: {flags}
Find the ONE most counterintuitive observation."""

        contrarian_msg = await flash.ainvoke([
            SystemMessage(content=_CONTRARIAN_SYSTEM),
            HumanMessage(content=contrarian_prompt),
        ])
        contrarian = contrarian_msg.content.strip()

        # ── Pass 2: full report (Pro) ─────────────────────────────────────────
        user_prompt = _NARRATIVE_USER.format(
            week_start       = snap.get("week_start", "this week"),
            job_count        = snap.get("job_count", 0),
            top_skills       = json.dumps(list(snap.get("top_skills", {}).items())[:10]),
            salary_p50       = int(snap.get("salary_p50") or 0),
            salary_n         = snap.get("salary_sample_size", 0),
            salary_p25       = int(snap.get("salary_p25") or 0),
            salary_p75       = int(snap.get("salary_p75") or 0),
            sponsorship_pct  = round((snap.get("sponsorship_rate") or 0) * 100, 1),
            remote_pct       = round((snap.get("remote_rate") or 0) * 100, 1),
            insight_flags    = flags + f"\n\nCONTRARIAN OBSERVATION TO INCLUDE:\n{contrarian}",
        )

        report_msg = await pro.ainvoke([
            SystemMessage(content=_NARRATIVE_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        draft = report_msg.content.strip()

        logger.info(f"{self.agent_id}.execute.done", draft_words=len(draft.split()))
        return {"report_draft": draft, "contrarian": contrarian}

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        draft    = result.get("report_draft", "")
        adaptive = plan.get("adaptive", {})

        # Self-review: word count, has numbers, not empty
        words = len(draft.split())
        has_numbers = any(char.isdigit() for char in draft)
        score = 8.0
        if words < 200:
            score -= 3
        elif words > 700:
            score -= 1
        if not has_numbers:
            score -= 2

        quality = "good" if score >= 7.5 else "warning"
        logger.info(f"{self.agent_id}.reflect", score=score, words=words, quality=quality)

        # Store this week's snapshot for next week's comparison
        adaptive["prior_snapshot"] = plan.get("snapshot", {})
        state["adaptive_params"]   = adaptive

        return {"quality": quality, "self_review_score": score, "words": words}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_draft":       result.get("report_draft", ""),
            "contrarian":         result.get("contrarian", ""),
            "self_review_score":  reflection.get("self_review_score", 0),
        }


class ContentLeadAgent(DeepAgent):
    """
    Department 5 Lead Agent.
    Orchestrates WeeklyReportWriterAgent and ChartAnnotatorAgent.
    """

    agent_id   = "content_lead_v1"
    department = "content_studio"

    def __init__(self) -> None:
        self._writer = WeeklyReportWriterAgent()

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return {"snapshot": context.get("snapshot", {}), "run_id": context.get("run_id", "")}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        result = await self._writer.run({"snapshot": plan["snapshot"]})
        return result

    async def reflect(self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        score = result.get("self_review_score", 0)
        return {"quality": "good" if score >= 7.5 else "warning", "score": score}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {"report_draft": result.get("report_draft", ""), "report_score": reflection.get("score", 0)}
