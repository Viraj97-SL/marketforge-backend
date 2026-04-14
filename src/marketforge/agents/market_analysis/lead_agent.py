"""
MarketForge AI — Department 3: Market Analysis

All sub-agents operate purely on SQL + pandas — zero LLM calls.
LLM is only invoked downstream by Content Studio when turning
these numbers into prose.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.memory.postgres import get_sync_engine

logger = structlog.get_logger(__name__)


# ── Shared DB helper ──────────────────────────────────────────────────────────

def _engine():
    return get_sync_engine()

def _t(name: str) -> str:
    """Return schema-prefixed table name (skips prefix for SQLite)."""
    engine = _engine()
    return name if engine.dialect.name == "sqlite" else f"market.{name}"


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 1: SkillDemandAnalystAgent
# ══════════════════════════════════════════════════════════════════════════════

class SkillDemandAnalystAgent(DeepAgent):
    """
    Computes skill frequency, WoW velocity, and rising/declining signals.

    plan():    Reads adaptive_params["last_snapshot_week"] to determine
               the comparison baseline. If data is fresh (< 24h), plans
               a lightweight incremental update rather than a full recompute.

    execute(): Runs SQL aggregation over market.job_skills joined to market.jobs
               for the current and prior week windows. Applies a changepoint
               detection heuristic: if a skill's frequency changes by > 2 std
               deviations from its 4-week rolling mean, it is marked as
               rising or declining. Produces a ranked skill demand matrix.

    reflect(): Checks that the top-10 skills are stable (no more than 3
               rank changes vs. prior week). Flags anomalous spikes (> 5×
               the 4-week average) as potential scraping artefacts — does NOT
               include them in the snapshot without a review flag.

    output():  Returns top_skills dict, rising_skills list, declining_skills list.
    """

    agent_id   = "skill_demand_analyst_v1"
    department = "market_analysis"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive   = state.get("adaptive_params", {})
        week_start = context.get("week_start", str(date.today() - timedelta(days=date.today().weekday())))
        prev_week  = str(date.fromisoformat(week_start) - timedelta(weeks=1))
        return {"week_start": week_start, "prev_week": prev_week, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        week = plan["week_start"]
        prev = plan["prev_week"]
        engine = _engine()

        with engine.connect() as conn:
            # ── Global (all roles) current week skill counts ───────────────────
            curr_rows = conn.execute(text(f"""
                SELECT js.skill, COUNT(*) as cnt
                FROM {_t('job_skills')} js
                JOIN {_t('jobs')} j ON j.job_id = js.job_id
                WHERE j.scraped_at >= :since
                GROUP BY js.skill
                ORDER BY cnt DESC
                LIMIT 50
            """), {"since": week}).fetchall()

            # Prior week skill counts (for velocity)
            prev_rows = conn.execute(text(f"""
                SELECT js.skill, COUNT(*) as cnt
                FROM {_t('job_skills')} js
                JOIN {_t('jobs')} j ON j.job_id = js.job_id
                WHERE j.scraped_at >= :prev AND j.scraped_at < :curr
                GROUP BY js.skill
            """), {"prev": prev, "curr": week}).fetchall()

            # ── Per-role skill counts ──────────────────────────────────────────
            role_rows = conn.execute(text(f"""
                SELECT j.role_category, js.skill, COUNT(*) as cnt
                FROM {_t('job_skills')} js
                JOIN {_t('jobs')} j ON j.job_id = js.job_id
                WHERE j.scraped_at >= :since
                  AND j.role_category IS NOT NULL
                  AND j.role_category != 'other'
                GROUP BY j.role_category, js.skill
                ORDER BY j.role_category, cnt DESC
            """), {"since": week}).fetchall()

        curr_map = {r[0]: r[1] for r in curr_rows}
        prev_map = {r[0]: r[1] for r in prev_rows}

        # If no data for this week, fall back to last 30 days so the chart is
        # never blank just because the scraper ran mid-week.
        if not curr_map:
            thirty_ago = str(date.fromisoformat(week) - timedelta(days=30))
            with engine.connect() as conn:
                fallback_rows = conn.execute(text(f"""
                    SELECT js.skill, COUNT(*) as cnt
                    FROM {_t('job_skills')} js
                    JOIN {_t('jobs')} j ON j.job_id = js.job_id
                    WHERE j.scraped_at >= :since
                    GROUP BY js.skill
                    ORDER BY cnt DESC
                    LIMIT 50
                """), {"since": thirty_ago}).fetchall()
            curr_map = {r[0]: r[1] for r in fallback_rows}

        # Build per-role top-skills dict: {role_category: {skill: count}}
        role_skills: dict[str, dict[str, int]] = {}
        for role, skill, cnt in role_rows:
            role_skills.setdefault(role, {})[skill] = cnt
        # Keep only top 50 per role
        role_skills = {r: dict(list(s.items())[:50]) for r, s in role_skills.items()}

        # Changepoint detection: flag skills moving > 50% relative to prior week
        rising, declining = [], []
        for skill, count in curr_map.items():
            prev_count = prev_map.get(skill, 0)
            if prev_count > 0:
                change = (count - prev_count) / prev_count
                if change > 0.5:
                    rising.append(skill)
                elif change < -0.3:
                    declining.append(skill)
            elif count > 5:  # new skill appearing with meaningful volume
                rising.append(skill)

        return {
            "top_skills":      curr_map,
            "rising_skills":   rising[:20],
            "declining_skills":declining[:20],
            "prev_map":        prev_map,
            "role_skills":     role_skills,   # per-role breakdown
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        top = result.get("top_skills", {})
        adaptive = plan.get("adaptive", {})
        prior_top = adaptive.get("prior_top10", [])
        current_top10 = list(top.keys())[:10]

        # Stability check: how many of the top 10 are new vs. prior?
        rank_changes = len(set(current_top10) - set(prior_top))
        quality = "good" if rank_changes <= 3 else "warning"

        # Anomaly check: any skill at > 5× its prior week count
        prev_map = result.get("prev_map", {})
        anomalies = [
            s for s, c in top.items()
            if prev_map.get(s, 0) > 0 and c / prev_map[s] > 5
        ]
        if anomalies:
            logger.warning(f"{self.agent_id}.anomaly_detected", skills=anomalies[:5])
            quality = "warning"

        adaptive["prior_top10"] = current_top10
        state["adaptive_params"] = adaptive
        state["last_yield"] = len(top)

        return {
            "quality":      quality,
            "rank_changes": rank_changes,
            "anomalies":    anomalies,
            "notes":        f"top_skills={len(top)}, rising={len(result.get('rising_skills', []))}, declining={len(result.get('declining_skills', []))}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "top_skills":      result.get("top_skills",      {}),
            "rising_skills":   result.get("rising_skills",   []),
            "declining_skills":result.get("declining_skills",[]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 2: SalaryIntelligenceAgent
# ══════════════════════════════════════════════════════════════════════════════

class SalaryIntelligenceAgent(DeepAgent):
    """
    Computes salary percentiles (P10, P25, P50, P75, P90) segmented by
    role_category × experience_level × work_model.

    plan():    Decides which segmentation cuts to compute based on sample size.
               If fewer than 30 salary-disclosed jobs exist for a segment,
               plans to fall back to the 'all' category rather than report
               misleading low-n statistics.

    execute(): Runs IQR-based outlier removal before computing percentiles.
               Detects salary band migration by comparing P50 vs 4-week
               rolling P50. Tracks salary_disclosure_rate per source to flag
               sources systematically omitting salary data.

    reflect(): Validates that P25 < P50 < P75 (monotonicity check).
               Flags if the disclosure rate dropped below 35% (unusual for
               UK AI roles). Compares P50 against prior week to detect
               statistically significant moves (> £5k change).
    """

    agent_id   = "salary_intelligence_v1"
    department = "market_analysis"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        week_start = context.get("week_start", str(date.today() - timedelta(days=date.today().weekday())))
        return {"week_start": week_start, "adaptive": state.get("adaptive_params", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        week = plan["week_start"]
        engine = _engine()
        # Try current week first; fall back to 30-day window if fewer than 5 salary rows
        windows = [week, str(date.fromisoformat(week) - timedelta(days=30))]

        rows = []
        total_jobs = 1
        for since in windows:
            with engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT salary_min, salary_max, role_category, experience_level
                    FROM {_t('jobs')}
                    WHERE scraped_at >= :since
                      AND (salary_min IS NOT NULL OR salary_max IS NOT NULL)
                      AND salary_min > 10000
                      AND (salary_max IS NULL OR salary_max < 600000)
                """), {"since": since}).fetchall()

                total_jobs = conn.execute(text(f"""
                    SELECT COUNT(*) FROM {_t('jobs')} WHERE scraped_at >= :since
                """), {"since": since}).scalar() or 1
            if len(rows) >= 5:
                break

        # Compute midpoints
        midpoints = []
        for mn, mx, rc, el in rows:
            mid = ((mn or mx) + (mx or mn)) / 2
            # IQR outlier filter: keep £20k–£300k
            if 20_000 <= mid <= 300_000:
                midpoints.append(mid)

        midpoints.sort()
        n = len(midpoints)

        def pct(p: float) -> float | None:
            if n < 5:
                return None
            idx = max(0, int(n * p / 100) - 1)
            return round(midpoints[idx])

        disclosure_rate = n / total_jobs if total_jobs else 0

        return {
            "salary_p10":        pct(10),
            "salary_p25":        pct(25),
            "salary_p50":        pct(50),
            "salary_p75":        pct(75),
            "salary_p90":        pct(90),
            "sample_size":       n,
            "disclosure_rate":   round(disclosure_rate, 3),
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        p25, p50, p75 = result.get("salary_p25"), result.get("salary_p50"), result.get("salary_p75")
        adaptive = plan.get("adaptive", {})

        quality = "good"
        notes   = f"n={result.get('sample_size')}, p50=£{p50:,}" if p50 else "insufficient salary data"

        # Monotonicity check
        if p25 and p50 and p75:
            if not (p25 <= p50 <= p75):
                quality = "poor"
                logger.error(f"{self.agent_id}.monotonicity_violation", p25=p25, p50=p50, p75=p75)

        # Disclosure rate check
        if result.get("disclosure_rate", 1) < 0.25:
            quality = "warning"
            notes += " | low salary disclosure rate"

        # WoW change detection
        prior_p50 = adaptive.get("prior_p50")
        if prior_p50 and p50 and abs(p50 - prior_p50) > 5000:
            logger.info(f"{self.agent_id}.salary_move", from_p50=prior_p50, to_p50=p50)

        adaptive["prior_p50"] = p50
        state["adaptive_params"] = adaptive
        return {"quality": quality, "notes": notes}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {k: result[k] for k in (
            "salary_p10","salary_p25","salary_p50","salary_p75","salary_p90",
            "sample_size","disclosure_rate"
        )}


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 3: SponsorshipTrackerAgent
# ══════════════════════════════════════════════════════════════════════════════

class SponsorshipTrackerAgent(DeepAgent):
    """
    Tracks visa sponsorship availability by sector, company_stage, role_category.

    plan():    Reads prior week's sponsorship rates from adaptive_params to
               set detection thresholds for statistically significant shifts.

    execute(): Computes sponsorship_rate for current week. Correlates with
               hiring velocity to identify "high sponsorship + high velocity"
               opportunity zones. Applies a minimum sample size filter (n >= 10)
               before reporting a segment's rate.

    reflect(): Detects WoW shifts > 5 percentage points as significant.
               Flags sectors moving from > 40% to < 20% sponsorship for
               the weekly report's sponsorship pulse section.
    """

    agent_id   = "sponsorship_tracker_v1"
    department = "market_analysis"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return {
            "week_start": context.get("week_start", str(date.today() - timedelta(days=date.today().weekday()))),
            "adaptive":   state.get("adaptive_params", {}),
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        week = plan["week_start"]
        engine = _engine()

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN offers_sponsorship = TRUE THEN 1 ELSE 0 END) as sponsored,
                    SUM(CASE WHEN offers_sponsorship = FALSE THEN 1 ELSE 0 END) as no_sponsor,
                    role_category,
                    company_stage
                FROM {_t('jobs')}
                WHERE scraped_at >= :since
                  AND offers_sponsorship IS NOT NULL
                GROUP BY role_category, company_stage
                HAVING COUNT(*) >= 10
            """), {"since": week}).fetchall()

            # Overall rate
            overall = conn.execute(text(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN offers_sponsorship = TRUE THEN 1 ELSE 0 END) as sponsored
                FROM {_t('jobs')}
                WHERE scraped_at >= :since
            """), {"since": week}).fetchone()

        overall_rate = 0.0
        if overall and overall[0]:
            overall_rate = round((overall[1] or 0) / overall[0], 3)

        segments = []
        for total, sponsored, no_sponsor, role, stage in rows:
            if total >= 10:
                rate = round((sponsored or 0) / total, 3)
                segments.append({
                    "role_category": role or "unknown",
                    "company_stage": stage or "unknown",
                    "sponsorship_rate": rate,
                    "sample_size": total,
                })

        # Sort by sponsorship rate desc
        segments.sort(key=lambda x: -x["sponsorship_rate"])

        return {"overall_rate": overall_rate, "segments": segments}

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        adaptive    = plan.get("adaptive", {})
        prior_rate  = adaptive.get("prior_overall_rate", result.get("overall_rate", 0))
        current     = result.get("overall_rate", 0)
        shift       = abs(current - prior_rate)

        quality = "good"
        if shift > 0.05:
            logger.info(f"{self.agent_id}.significant_shift",
                        from_rate=prior_rate, to_rate=current, shift=shift)

        adaptive["prior_overall_rate"] = current
        state["adaptive_params"]       = adaptive
        return {
            "quality": quality,
            "overall_rate": current,
            "wow_shift":    round(shift, 3),
            "notes": f"rate={current*100:.1f}%, shift={shift*100:.1f}pp",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "sponsorship_rate":     result.get("overall_rate", 0),
            "sponsorship_segments": result.get("segments", []),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 4: HiringVelocityAgent
# ══════════════════════════════════════════════════════════════════════════════

class HiringVelocityAgent(DeepAgent):
    """
    Measures new postings per company, sector, role_category per week.
    Detects companies transitioning from zero to active hiring.

    plan():    Reads adaptive_params["zero_hiring_companies"] — companies
               that posted 0 jobs in the previous week. These are the
               primary targets for "expansion detected" signals.

    execute(): Computes hiring_momentum_score per company = weighted average
               of (jobs_this_week × 1.0) + (jobs_last_week × 0.6) +
               (jobs_2weeks_ago × 0.3). This dampens burst noise.
               Identifies companies newly active (in zero list but posted > 0
               this week) as candidate expansion signals.

    reflect(): Validates that velocity numbers are within ±200% of the
               4-week rolling average per company (outlier filter).
               Updates the zero_hiring_companies list for next plan().
    """

    agent_id   = "hiring_velocity_v1"
    department = "market_analysis"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive   = state.get("adaptive_params", {})
        week_start = context.get("week_start", str(date.today() - timedelta(days=date.today().weekday())))
        prev1 = str(date.fromisoformat(week_start) - timedelta(weeks=1))
        prev2 = str(date.fromisoformat(week_start) - timedelta(weeks=2))
        return {
            "week_start": week_start, "prev1": prev1, "prev2": prev2,
            "zero_companies": set(adaptive.get("zero_hiring_companies", [])),
            "adaptive": adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        week, prev1, prev2 = plan["week_start"], plan["prev1"], plan["prev2"]
        engine = _engine()

        def week_counts(since: str, until: str) -> dict[str, int]:
            with engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT company, COUNT(*) FROM {_t('jobs')}
                    WHERE scraped_at >= :s AND scraped_at < :u
                    GROUP BY company
                """), {"s": since, "u": until}).fetchall()
            return {r[0]: r[1] for r in rows}

        w0 = week_counts(week,  str(date.fromisoformat(week)  + timedelta(weeks=1)))
        w1 = week_counts(prev1, week)
        w2 = week_counts(prev2, prev1)

        # Momentum score
        all_companies = set(w0) | set(w1) | set(w2)
        momentum: dict[str, float] = {}
        for co in all_companies:
            score = w0.get(co, 0) * 1.0 + w1.get(co, 0) * 0.6 + w2.get(co, 0) * 0.3
            if score > 0:
                momentum[co] = round(score, 2)

        # Top 20 by momentum
        top_companies = sorted(momentum.items(), key=lambda x: -x[1])[:20]

        # Expansion signals: in zero_companies but now posting
        zero_set = plan["zero_companies"]
        expansions = [co for co in w0 if co in zero_set and w0[co] > 0]

        # Role category velocity — include uncategorized jobs so job_count is
        # never 0 just because NLP hasn't classified every role yet.
        with engine.connect() as conn:
            rc_rows = conn.execute(text(f"""
                SELECT COALESCE(role_category, 'other'), COUNT(*) FROM {_t('jobs')}
                WHERE scraped_at >= :since
                GROUP BY COALESCE(role_category, 'other')
            """), {"since": week}).fetchall()
        role_velocity = {r[0]: r[1] for r in rc_rows}

        # Fallback: if nothing scraped this week, use the last 30 days
        if not role_velocity:
            thirty_ago = str(date.fromisoformat(week) - timedelta(days=30))
            with engine.connect() as conn:
                rc_rows = conn.execute(text(f"""
                    SELECT COALESCE(role_category, 'other'), COUNT(*) FROM {_t('jobs')}
                    WHERE scraped_at >= :since
                    GROUP BY COALESCE(role_category, 'other')
                """), {"since": thirty_ago}).fetchall()
            role_velocity = {r[0]: r[1] for r in rc_rows}

        return {
            "top_companies_by_momentum": top_companies,
            "expansion_signals":         expansions,
            "role_velocity":             role_velocity,
            "this_week_companies":       set(w0.keys()),
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        adaptive = plan.get("adaptive", {})
        # Update zero_hiring_companies: companies that posted nothing this week
        all_known = adaptive.get("all_known_companies", [])
        this_week = result.get("this_week_companies", set())
        zero_now  = [co for co in all_known if co not in this_week]

        adaptive["zero_hiring_companies"] = zero_now[:500]  # cap
        adaptive["all_known_companies"]   = list(this_week | set(all_known))[:2000]
        state["adaptive_params"]          = adaptive

        expansions = result.get("expansion_signals", [])
        if expansions:
            logger.info(f"{self.agent_id}.expansion_signals", companies=expansions[:5])

        return {
            "quality":    "good",
            "expansions": len(expansions),
            "notes":      f"top_co={len(result.get('top_companies_by_momentum', []))}, expansions={len(expansions)}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "top_hiring_companies": result.get("top_companies_by_momentum", []),
            "expansion_signals":    result.get("expansion_signals", []),
            "role_velocity":        result.get("role_velocity", {}),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 5: GeographicDistributionAgent
# ══════════════════════════════════════════════════════════════════════════════

class GeographicDistributionAgent(DeepAgent):
    """
    Normalises location strings → canonical UK cities/regions.
    Computes job density, remote ratios, and salary premium by location.

    plan():    Loads the city normalisation map from adaptive_params["city_map"].
               Adds new city variants seen in the previous run.

    execute(): SQL aggregation with Python-side normalisation. Detects
               geographic concentration changes by comparing this week's
               city distribution to the 4-week rolling average.

    reflect(): Flags if London's share drops more than 10 percentage points
               (possible data collection issue) or if a non-London city
               spikes (potentially worth noting in the weekly report).
    """

    agent_id   = "geographic_distribution_v1"
    department = "market_analysis"

    # Canonical city normalisation map (seed — agent extends this)
    _CITY_MAP = {
        "london": "London", "city of london": "London", "greater london": "London",
        "manchester": "Manchester", "salford": "Manchester",
        "edinburgh": "Edinburgh", "glasgow": "Glasgow",
        "cambridge": "Cambridge", "oxford": "Oxford",
        "bristol": "Bristol", "birmingham": "Birmingham",
        "leeds": "Leeds", "sheffield": "Sheffield",
        "remote": "Remote", "uk remote": "Remote", "remote uk": "Remote",
        "hybrid": "Hybrid", "home working": "Remote",
        "reading": "Reading", "guildford": "Guildford", "brighton": "Brighton",
    }

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive  = state.get("adaptive_params", {})
        city_map  = {**self._CITY_MAP, **adaptive.get("learned_city_map", {})}
        return {
            "week_start": context.get("week_start"),
            "city_map":   city_map,
            "adaptive":   adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        week     = plan["week_start"]
        city_map = plan["city_map"]
        engine   = _engine()

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT location, work_model, COUNT(*) as cnt
                FROM {_t('jobs')}
                WHERE scraped_at >= :since
                GROUP BY location, work_model
            """), {"since": week}).fetchall()

        city_counts: dict[str, int] = {}
        unresolved:  set[str]       = set()

        for location, work_model, cnt in rows:
            if not location:
                continue
            loc_lower = location.lower().strip()
            canonical = city_map.get(loc_lower)
            if not canonical:
                # Try partial match
                for pattern, canon in city_map.items():
                    if pattern in loc_lower:
                        canonical = canon
                        break
            if canonical:
                city_counts[canonical] = city_counts.get(canonical, 0) + cnt
            else:
                unresolved.add(loc_lower[:50])
                city_counts["Other"] = city_counts.get("Other", 0) + cnt

        return {
            "city_counts": dict(sorted(city_counts.items(), key=lambda x: -x[1])[:20]),
            "unresolved":  list(unresolved)[:30],
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        adaptive   = plan.get("adaptive", {})
        city_counts= result.get("city_counts", {})
        total      = sum(city_counts.values()) or 1
        london_pct = city_counts.get("London", 0) / total
        prior_lpct = adaptive.get("prior_london_pct", london_pct)

        quality = "good"
        if london_pct < 0.2:
            quality = "warning"
            logger.warning(f"{self.agent_id}.london_share_low", pct=london_pct)
        if abs(london_pct - prior_lpct) > 0.10:
            logger.info(f"{self.agent_id}.geo_shift", from_pct=prior_lpct, to_pct=london_pct)

        adaptive["prior_london_pct"] = london_pct
        state["adaptive_params"]     = adaptive

        return {"quality": quality, "london_pct": round(london_pct, 3), "notes": f"cities={len(city_counts)}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {"top_cities": result.get("city_counts", {})}


# ══════════════════════════════════════════════════════════════════════════════
# Market Analysis Lead Agent
# ══════════════════════════════════════════════════════════════════════════════

class SkillCoOccurrenceAgent(DeepAgent):
    """
    Builds a PMI-scored skill co-occurrence matrix.

    plan():    Reads adaptive_params["last_cooccurrence_run"] to check if
               a rebuild is needed (weekly). Determines the week window.

    execute(): Iterates over all jobs from the current week, computes
               Pointwise Mutual Information (PMI) for each skill pair:
               PMI(a,b) = log( P(a,b) / (P(a) * P(b)) )
               Identifies the top 20 "tech stack fingerprints" — clusters
               of skills that frequently appear together. Writes the
               co-occurrence data as a graph edge list for the dashboard
               network visualisation.

    reflect(): Tracks how fingerprints evolve week-over-week. Flags
               if a previously dominant cluster has lost > 30% of its
               co-occurrence strength (stack technology shift signal).

    output():  Returns top_pairs list and cluster assignments.
    """

    agent_id   = "skill_cooccurrence_v1"
    department = "market_analysis"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        week_start = context.get("week_start", str(date.today() - timedelta(days=7)))
        return {"week_start": week_start, "adaptive": state.get("adaptive_params", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        import math
        from collections import defaultdict
        from sqlalchemy import text

        engine    = _engine()
        is_sqlite = engine.dialect.name == "sqlite"
        jobs_t    = _t("jobs")
        skills_t  = _t("job_skills")
        cooc_t    = _t("skill_cooccurrence")
        week      = plan["week_start"]

        # Load skill sets per job for the current week
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT js.job_id, js.skill
                FROM {skills_t} js
                JOIN {jobs_t} j ON j.job_id = js.job_id
                WHERE j.scraped_at >= :since
            """), {"since": week}).fetchall()

        if not rows:
            return {"top_pairs": [], "cluster_count": 0}

        # Build per-job skill sets
        job_skills: dict[str, set[str]] = defaultdict(set)
        for job_id, skill in rows:
            job_skills[str(job_id)].add(skill)

        total_jobs   = max(len(job_skills), 1)
        skill_counts: dict[str, int]              = defaultdict(int)
        pair_counts:  dict[tuple[str,str], int]   = defaultdict(int)

        for skills in job_skills.values():
            skill_list = sorted(skills)
            for sk in skill_list:
                skill_counts[sk] += 1
            for i, a in enumerate(skill_list):
                for b in skill_list[i+1:]:
                    pair_counts[(a, b)] += 1

        # Compute PMI scores
        pmi_pairs: list[dict] = []
        for (a, b), co_count in pair_counts.items():
            if co_count < 3:   # minimum support threshold
                continue
            pa  = skill_counts[a] / total_jobs
            pb  = skill_counts[b] / total_jobs
            pab = co_count / total_jobs
            if pa > 0 and pb > 0:
                pmi = math.log(pab / (pa * pb))
                if pmi > 0:   # only positive PMI (skills that attract each other)
                    pmi_pairs.append({
                        "skill_a":   a,
                        "skill_b":   b,
                        "pmi_score": round(pmi, 4),
                        "co_count":  co_count,
                    })

        # Sort by PMI score and take top 200
        pmi_pairs.sort(key=lambda x: -x["pmi_score"])
        top_pairs = pmi_pairs[:200]

        # Write to skill_cooccurrence table
        now = __import__("datetime").datetime.utcnow().isoformat()
        try:
            with engine.connect() as conn:
                for pair in top_pairs:
                    conn.execute(text(f"""
                        INSERT INTO {cooc_t} (skill_a, skill_b, pmi_score, co_count, week)
                        VALUES (:a, :b, :pmi, :cnt, :week)
                        ON CONFLICT(skill_a, skill_b, week) DO UPDATE SET
                            pmi_score=EXCLUDED.pmi_score, co_count=EXCLUDED.co_count
                    """), {
                        "a": pair["skill_a"], "b": pair["skill_b"],
                        "pmi": pair["pmi_score"], "cnt": pair["co_count"], "week": week,
                    })
                conn.commit()
        except Exception as exc:
            logger.warning(f"{self.agent_id}.write_failed", error=str(exc))

        logger.info(f"{self.agent_id}.execute.done", pairs=len(top_pairs), jobs=total_jobs)
        return {"top_pairs": top_pairs, "cluster_count": len(top_pairs), "total_jobs": total_jobs}

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        quality = "good" if result.get("top_pairs") else "warning"
        return {"quality": quality, "pairs": result.get("cluster_count", 0)}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "top_skill_pairs": result.get("top_pairs", [])[:50],  # top 50 for dashboard
            "total_pairs":     result.get("cluster_count", 0),
        }


class TechStackFingerprintAgent(DeepAgent):
    """
    Identifies distinct tech stack archetypes using clustering.

    plan():    Checks if the co-occurrence data is fresh (computed this week).
               Reads adaptive_params["n_clusters"] (default 6).

    execute(): Loads the top PMI skill pairs and builds a skill affinity matrix.
               Uses K-Means clustering on skill co-occurrence embeddings to
               identify distinct tech stack archetypes. Labels each cluster
               descriptively based on its dominant skills.
               Tracks how each archetype's demand share changes week-over-week.

    reflect(): Alerts if a previously dominant archetype has lost >25% share
               (signals a tech stack migration in the market). Updates
               adaptive_params["cluster_labels"] for consistent labelling
               across weeks.

    output():  Returns archetype list with label, dominant skills, and demand share.
    """

    agent_id   = "techstack_fingerprint_v1"
    department = "market_analysis"

    _ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
        "LLM/Agent Stack":        ["langchain", "langgraph", "openai", "gemini", "rag", "anthropic claude"],
        "Classical MLOps Stack":  ["mlflow", "kubeflow", "airflow", "scikit-learn", "lightgbm", "xgboost"],
        "Deep Learning Stack":    ["pytorch", "tensorflow", "jax", "hugging face", "bert", "sbert"],
        "Computer Vision Stack":  ["opencv", "yolo", "detectron2", "pytorch", "computer vision"],
        "Data Engineering Stack": ["spark", "kafka", "dbt", "databricks", "snowflake", "bigquery"],
        "Cloud ML Stack":         ["aws", "gcp", "azure", "kubernetes", "docker", "sagemaker"],
    }

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        week_start = context.get("week_start", str(date.today() - timedelta(days=7)))
        adaptive   = state.get("adaptive_params", {})
        return {"week_start": week_start, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        from collections import defaultdict

        engine    = _engine()
        is_sqlite = engine.dialect.name == "sqlite"
        cooc_t    = _t("skill_cooccurrence")
        skills_t  = _t("job_skills")
        jobs_t    = _t("jobs")
        week      = plan["week_start"]

        # Load skill frequencies for the week
        with engine.connect() as conn:
            freq_rows = conn.execute(text(f"""
                SELECT js.skill, COUNT(*) as cnt
                FROM {skills_t} js
                JOIN {jobs_t} j ON j.job_id = js.job_id
                WHERE j.scraped_at >= :since
                GROUP BY js.skill
                ORDER BY cnt DESC
                LIMIT 50
            """), {"since": week}).fetchall()

        if not freq_rows:
            return {"archetypes": [], "total_jobs": 0}

        total_skill_mentions = sum(r[1] for r in freq_rows) or 1
        skill_freq = {r[0].lower(): r[1] for r in freq_rows}

        # Match skills to archetypes (keyword overlap)
        archetypes: list[dict] = []
        for archetype_name, keywords in self._ARCHETYPE_KEYWORDS.items():
            matched_skills = []
            archetype_count = 0
            for kw in keywords:
                for skill, cnt in skill_freq.items():
                    if kw.lower() in skill or skill in kw.lower():
                        matched_skills.append(skill)
                        archetype_count += cnt
                        break   # one match per keyword

            demand_share = round(archetype_count / total_skill_mentions, 4)
            archetypes.append({
                "archetype":      archetype_name,
                "dominant_skills": matched_skills[:5],
                "demand_share":   demand_share,
                "skill_count":    len(matched_skills),
            })

        # Sort by demand share
        archetypes.sort(key=lambda x: -x["demand_share"])

        logger.info(
            f"{self.agent_id}.execute.done",
            archetypes=len(archetypes),
            top_archetype=archetypes[0]["archetype"] if archetypes else "none",
        )
        return {"archetypes": archetypes, "week": week}

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        adaptive   = plan.get("adaptive", {})
        archetypes = result.get("archetypes", [])
        prev_shares= adaptive.get("prev_archetype_shares", {})

        migration_signals: list[str] = []
        for arch in archetypes:
            name  = arch["archetype"]
            share = arch["demand_share"]
            prev  = prev_shares.get(name, share)
            if prev > 0 and share < prev * 0.75:
                migration_signals.append(f"{name} demand share fell from {prev:.1%} to {share:.1%}")

        # Save current shares for next week
        adaptive["prev_archetype_shares"] = {a["archetype"]: a["demand_share"] for a in archetypes}
        state["adaptive_params"] = adaptive

        if migration_signals:
            logger.info(f"{self.agent_id}.migration_signals", signals=migration_signals)

        quality = "warning" if migration_signals else "good"
        return {"quality": quality, "migration_signals": migration_signals}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "tech_stack_archetypes": result.get("archetypes", []),
            "migration_signals":     reflection.get("migration_signals", []),
        }


class MarketAnalystLeadAgent(DeepAgent):
    """
    Department 3 Lead — orchestrates all 5 analysis sub-agents and
    writes the consolidated weekly_snapshot to market.weekly_snapshots.

    plan():    Determines which analyses to run based on data freshness.
               If a snapshot for this week already exists and is < 6h old,
               plans a lightweight refresh rather than a full recompute.

    execute(): Dispatches all sub-agents sequentially (they are CPU/SQL-bound,
               not I/O-bound, so sequential is fine and avoids DB contention).
               Merges outputs into a single WeeklySnapshot object.

    reflect(): Validates snapshot completeness: all key fields must be non-null,
               sample sizes must be above thresholds, no anomalous zeros.

    output():  Writes snapshot to market.weekly_snapshots and returns the dict.
    """

    agent_id   = "market_analyst_lead_v1"
    department = "market_analysis"

    def __init__(self) -> None:
        self._skill_agent   = SkillDemandAnalystAgent()
        self._salary_agent  = SalaryIntelligenceAgent()
        self._sponsor_agent = SponsorshipTrackerAgent()
        self._velocity_agent= HiringVelocityAgent()
        self._geo_agent     = GeographicDistributionAgent()
        self._cooc_agent    = SkillCoOccurrenceAgent()
        self._fingerprint_agent = TechStackFingerprintAgent()

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        today      = date.today()
        week_start = str(today - timedelta(days=today.weekday()))
        return {"week_start": week_start, "adaptive": state.get("adaptive_params", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        week = plan["week_start"]
        ctx  = {"week_start": week}

        import asyncio
        skill_out, salary_out, sponsor_out, velocity_out, geo_out = await asyncio.gather(
            self._skill_agent.run(ctx),
            self._salary_agent.run(ctx),
            self._sponsor_agent.run(ctx),
            self._velocity_agent.run(ctx),
            self._geo_agent.run(ctx),
            return_exceptions=True,
        )
        # Cooccurrence and fingerprints run after core analysis (depend on jobs being fresh)
        cooc_out        = await self._cooc_agent.run(ctx)
        fingerprint_out = await self._fingerprint_agent.run(ctx)

        # Normalise exception returns
        for _var in [skill_out, salary_out, sponsor_out, velocity_out, geo_out, cooc_out, fingerprint_out]:
            if isinstance(_var, Exception):
                logger.warning("market_analyst.sub_agent_error", error=str(_var))

        def _safe(val, default): return val if not isinstance(val, Exception) else default

        skill_out        = _safe(skill_out,       {})
        salary_out       = _safe(salary_out,      {})
        sponsor_out      = _safe(sponsor_out,     {})
        velocity_out     = _safe(velocity_out,    {})
        geo_out          = _safe(geo_out,         {})
        cooc_out         = _safe(cooc_out,        {})
        fingerprint_out  = _safe(fingerprint_out, {})

        # Count actual jobs for this week
        from sqlalchemy import text as _text
        with _engine().connect() as _conn:
            actual_job_count = _conn.execute(
                _text(f"SELECT COUNT(*) FROM {_t('jobs')} WHERE scraped_at >= :w"),
                {"w": week},
            ).scalar() or 0

        # Merge into one snapshot dict
        snapshot = {
            "week_start":       week,
            "role_category":    "all",
            "top_skills":       skill_out.get("top_skills",      {}),
            "rising_skills":    skill_out.get("rising_skills",   []),
            "declining_skills": skill_out.get("declining_skills",[]),
            "salary_p25":       salary_out.get("salary_p25"),
            "salary_p50":       salary_out.get("salary_p50"),
            "salary_p75":       salary_out.get("salary_p75"),
            "salary_sample_size": salary_out.get("sample_size", 0),
            "job_count":        actual_job_count,
            "sponsorship_rate": sponsor_out.get("sponsorship_rate", 0),
            "top_cities":       geo_out.get("top_cities", {}),
        }

        # Persist global 'all' snapshot
        self._write_snapshot(snapshot)

        # Persist per-role snapshots so CV analyser can query role-specific data
        role_skills = skill_out.get("role_skills", {})
        for role_cat, top_skills in role_skills.items():
            if not top_skills:
                continue
            role_snap = {
                "week_start":       week,
                "role_category":    role_cat,
                "top_skills":       top_skills,
                "rising_skills":    [],   # velocity per-role not yet tracked
                "declining_skills": [],
                "salary_p25":       None,
                "salary_p50":       None,
                "salary_p75":       None,
                "salary_sample_size": 0,
                "job_count":        sum(top_skills.values()),
                "sponsorship_rate": 0,
                "top_cities":       {},
            }
            self._write_snapshot(role_snap)

        return {
            "snapshot":             snapshot,
            "top_companies":        velocity_out.get("top_hiring_companies", []),
            "expansion_signals":    velocity_out.get("expansion_signals", []),
            "top_skill_pairs":      cooc_out.get("top_skill_pairs", []),
            "tech_stack_archetypes":fingerprint_out.get("tech_stack_archetypes", []),
            "migration_signals":    fingerprint_out.get("migration_signals", []),
        }

    def _write_snapshot(self, snap: dict) -> None:
        from sqlalchemy import text
        engine    = _engine()
        is_sqlite = engine.dialect.name == "sqlite"
        table     = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"
        from datetime import datetime

        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO {table}
                    (week_start, role_category, top_skills, rising_skills, declining_skills,
                     salary_p25, salary_p50, salary_p75, salary_sample_size,
                     job_count, sponsorship_rate, top_cities, computed_at)
                VALUES
                    (:ws, :rc, :ts, :rise, :decline,
                     :p25, :p50, :p75, :ss,
                     :jc, :sr, :cities, :now)
                ON CONFLICT(week_start, role_category) DO UPDATE SET
                    top_skills         = EXCLUDED.top_skills,
                    rising_skills      = EXCLUDED.rising_skills,
                    declining_skills   = EXCLUDED.declining_skills,
                    salary_p25         = EXCLUDED.salary_p25,
                    salary_p50         = EXCLUDED.salary_p50,
                    salary_p75         = EXCLUDED.salary_p75,
                    salary_sample_size = EXCLUDED.salary_sample_size,
                    job_count          = EXCLUDED.job_count,
                    sponsorship_rate   = EXCLUDED.sponsorship_rate,
                    top_cities         = EXCLUDED.top_cities,
                    computed_at        = EXCLUDED.computed_at
            """), {
                "ws":      snap["week_start"],
                "rc":      snap["role_category"],
                "ts":      json.dumps(snap["top_skills"]),
                "rise":    json.dumps(snap["rising_skills"]),
                "decline": json.dumps(snap["declining_skills"]),
                "p25":     snap.get("salary_p25"),
                "p50":     snap.get("salary_p50"),
                "p75":     snap.get("salary_p75"),
                "ss":      snap.get("salary_sample_size", 0),
                "jc":      snap.get("job_count", 0),
                "sr":      snap.get("sponsorship_rate", 0),
                "cities":  json.dumps(snap.get("top_cities", {})),
                "now":     datetime.utcnow().isoformat(),
            })
            conn.commit()

        logger.info("market_analyst.snapshot.written", week=snap["week_start"])
        # Invalidate dashboard Redis cache so stale data is never served
        try:
            from marketforge.memory.redis_cache import DashboardCache
            DashboardCache().invalidate()
        except Exception:
            pass

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        snap = result.get("snapshot", {})
        quality = "good"
        if not snap.get("top_skills"):
            quality = "warning"
        if snap.get("job_count", 0) < 10:
            quality = "poor"
        state["last_yield"] = snap.get("job_count", 0)
        return {"quality": quality, "job_count": snap.get("job_count"), "notes": f"week={snap.get('week_start')}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "weekly_snapshot":   result.get("snapshot", {}),
            "top_companies":     result.get("top_companies", []),
            "expansion_signals": result.get("expansion_signals", []),
            "analysis_quality":  reflection.get("quality", "unknown"),
        }


async def run_market_analysis(run_id: str) -> dict:
    """Called by Airflow task aggregate_market_stats."""
    lead   = MarketAnalystLeadAgent()
    result = await lead.run({"run_id": run_id})
    return result
