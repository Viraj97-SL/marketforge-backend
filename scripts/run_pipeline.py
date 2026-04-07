"""
MarketForge AI — Manual Pipeline Runner

Runs the full ingestion pipeline outside Airflow (useful for local testing,
backfills, and ad-hoc runs).

Usage:
    python scripts/run_pipeline.py                   # Full ingestion run
    python scripts/run_pipeline.py --scrape-only     # Data collection only
    python scripts/run_pipeline.py --skip-nlp        # Skip NLP extraction
    python scripts/run_pipeline.py --dry-run         # Print plan, no execution
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketforge.memory.postgres import init_database, PipelineRunStore
from marketforge.utils.logger import setup_logging
from marketforge.utils.cost_tracker import CostTracker


async def run_ingestion(scrape_only: bool = False, skip_nlp: bool = False, dry_run: bool = False) -> dict:
    import structlog
    log = structlog.get_logger("run_pipeline")

    run_id       = f"manual_{uuid.uuid4().hex[:8]}"
    cost_tracker = CostTracker(run_id=run_id)
    run_store    = PipelineRunStore()

    log.info("pipeline.start", run_id=run_id, dry_run=dry_run)

    if dry_run:
        log.info("pipeline.dry_run.plan")
        from marketforge.agents.data_collection.lead_agent import DataCollectionLeadAgent
        lead  = DataCollectionLeadAgent()
        from marketforge.memory.postgres import AgentStateStore
        store = AgentStateStore()
        state = store.load(lead.agent_id, lead.department)
        plan  = await lead.plan({}, state)
        log.info("pipeline.dry_run.plan_complete",
                 active_sources=plan.get("active_agents", []))
        return {"dry_run": True, "plan": plan}

    run_store.start(run_id, "manual_run")

    try:
        # ── Data Collection ───────────────────────────────────────────────────
        from marketforge.agents.data_collection.lead_agent import run_data_collection
        summary = await run_data_collection(run_id, cost_tracker)
        log.info("collection.done", **summary)

        if not scrape_only and not skip_nlp:
            # ── NLP Extraction — process all jobs not yet in job_skills ────────
            from marketforge.nlp.taxonomy import extract_skills_flat
            from marketforge.memory.postgres import get_sync_engine, JobStore
            from sqlalchemy import text

            engine    = get_sync_engine()
            is_sqlite = engine.dialect.name == "sqlite"
            jobs_t    = "jobs" if is_sqlite else "market.jobs"
            skills_t  = "job_skills" if is_sqlite else "market.job_skills"
            job_store = JobStore()

            with engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT j.job_id, j.title, j.description FROM {jobs_t} j
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {skills_t} s WHERE s.job_id = j.job_id
                    )
                """)).fetchall()

            log.info("nlp.start", jobs=len(rows))

            nlp_stats = {"gate1": 0, "gate2": 0, "gate3": 0}
            for job_id, title, desc in rows:
                skills = extract_skills_flat(f"{title} {desc or ''}")
                job_store.upsert_skills(job_id, [(s, c, m, cf) for s, c, m, cf in skills])
                for _, _, method, _ in skills:
                    nlp_stats[method] = nlp_stats.get(method, 0) + 1

            log.info("nlp.done", **nlp_stats)

        # ── Market Analysis (populate weekly_snapshots) ───────────────────────
        from marketforge.agents.market_analysis.lead_agent import run_market_analysis
        log.info("market_analysis.start")
        try:
            analysis_result = await run_market_analysis(run_id)
            log.info("market_analysis.done", **{k: v for k, v in analysis_result.items() if not isinstance(v, (dict, list))})
        except Exception as exc:
            log.warning("market_analysis.failed", error=str(exc))

        # ── Finish ────────────────────────────────────────────────────────────
        run_store.finish(run_id, "success",
                         jobs_scraped=summary.get("jobs_raw", 0),
                         jobs_new=summary.get("jobs_new", 0),
                         llm_cost_usd=cost_tracker.total_usd)

        cost_tracker.persist()
        log.info("pipeline.complete", cost_summary=cost_tracker.summary)
        return summary

    except Exception as exc:
        import traceback
        log.error("pipeline.fatal", error=str(exc), tb=traceback.format_exc())
        run_store.finish(run_id, "failed")
        raise


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="MarketForge AI — manual pipeline runner")
    parser.add_argument("--scrape-only", action="store_true",  help="Run data collection only")
    parser.add_argument("--skip-nlp",    action="store_true",  help="Skip NLP extraction step")
    parser.add_argument("--dry-run",     action="store_true",  help="Print plan without executing")
    args = parser.parse_args()

    init_database()
    result = asyncio.run(run_ingestion(
        scrape_only=args.scrape_only,
        skip_nlp=args.skip_nlp,
        dry_run=args.dry_run,
    ))

    print("\n── Pipeline Summary ──────────────────────────")
    for k, v in result.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")
    print("─" * 46)


if __name__ == "__main__":
    main()
