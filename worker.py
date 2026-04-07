"""
MarketForge AI — Production Pipeline Worker
Replaces Apache Airflow for Railway deployment.

Runs as a long-lived process with APScheduler executing the same cron schedules
that the Airflow DAGs use. Zero Airflow overhead — uses the same underlying
agent code via scripts/run_pipeline.py.

Schedule (UTC):
  Tue + Thu 07:00  — full ingestion (data collection + NLP + market analysis)
  Mon       07:00  — weekly analysis only (snapshot + report, no scrape)
  Sun       02:00  — model retrain
  Every 6h         — Redis cache refresh

Usage:
    python worker.py                  # Start the scheduler (blocks forever)
    python worker.py --run-now ingest # Trigger a job immediately and exit
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent / "src"))

import structlog
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

log = structlog.get_logger("worker")


# ── Job functions ──────────────────────────────────────────────────────────────

def job_ingest() -> None:
    """Full pipeline: scrape → NLP → market analysis → cache invalidation."""
    from marketforge.memory.postgres import init_database, PipelineRunStore
    from marketforge.utils.logger import setup_logging
    from marketforge.utils.cost_tracker import CostTracker
    from marketforge.memory.redis_cache import DashboardCache
    from sqlalchemy import text

    setup_logging()
    run_id = f"worker_{uuid.uuid4().hex[:8]}"
    log.info("worker.ingest.start", run_id=run_id)

    try:
        init_database()
        cost_tracker = CostTracker(run_id=run_id)
        run_store    = PipelineRunStore()
        run_store.start(run_id, "worker_ingest")

        # ── Data collection ───────────────────────────────────────────────────
        from marketforge.agents.data_collection.lead_agent import run_data_collection
        summary = asyncio.run(run_data_collection(run_id, cost_tracker))
        log.info("worker.ingest.collection_done", **summary)

        # ── NLP extraction ────────────────────────────────────────────────────
        from marketforge.nlp.taxonomy import extract_skills_flat
        from marketforge.memory.postgres import get_sync_engine, JobStore

        engine   = get_sync_engine()
        jobs_t   = "jobs" if engine.dialect.name == "sqlite" else "market.jobs"
        skills_t = "job_skills" if engine.dialect.name == "sqlite" else "market.job_skills"

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT j.job_id, j.title, j.description FROM {jobs_t} j
                WHERE NOT EXISTS (
                    SELECT 1 FROM {skills_t} s WHERE s.job_id = j.job_id
                )
            """)).fetchall()

        log.info("worker.ingest.nlp_start", jobs=len(rows))
        nlp_stats = {"gate1": 0, "gate2": 0, "gate3": 0}
        job_store = JobStore()

        for row in rows:
            job_id, title, description = row
            text_blob = f"{title}\n{description or ''}"
            result    = extract_skills_flat(text_blob)
            if result.get("skills"):
                skills = [(s, result.get("categories", {}).get(s, "general"),
                           result.get("method", "gate1"), 1.0)
                          for s in result["skills"]]
                job_store.upsert_skills(job_id, skills)
                gate = result.get("method", "gate1")
                nlp_stats[gate] = nlp_stats.get(gate, 0) + len(skills)

        log.info("worker.ingest.nlp_done", **nlp_stats)

        # ── Market analysis ───────────────────────────────────────────────────
        from marketforge.agents.market_analysis.lead_agent import MarketAnalystLeadAgent
        analyst = MarketAnalystLeadAgent()
        asyncio.run(analyst.execute({}, None))
        log.info("worker.ingest.analysis_done")

        # ── Cache invalidation ────────────────────────────────────────────────
        try:
            DashboardCache().invalidate()
            log.info("worker.ingest.cache_invalidated")
        except Exception as cache_exc:
            log.warning("worker.ingest.cache_skip", error=str(cache_exc))

        run_store.complete(run_id, status="success",
                           jobs_scraped=summary.get("jobs_new", 0),
                           jobs_new=summary.get("jobs_new", 0),
                           llm_cost_usd=cost_tracker.total_cost)
        log.info("worker.ingest.done", run_id=run_id)

    except Exception as exc:
        log.error("worker.ingest.failed", run_id=run_id, error=str(exc))
        raise


def job_weekly_analysis() -> None:
    """Generate weekly snapshot + report (no scraping)."""
    from marketforge.utils.logger import setup_logging
    setup_logging()
    run_id = f"worker_analysis_{uuid.uuid4().hex[:8]}"
    log.info("worker.analysis.start", run_id=run_id)

    try:
        from marketforge.agents.market_analysis.lead_agent import MarketAnalystLeadAgent
        analyst = MarketAnalystLeadAgent()
        asyncio.run(analyst.execute({}, None))

        from marketforge.memory.redis_cache import DashboardCache
        try:
            DashboardCache().invalidate()
        except Exception:
            pass

        log.info("worker.analysis.done", run_id=run_id)
    except Exception as exc:
        log.error("worker.analysis.failed", run_id=run_id, error=str(exc))
        raise


def job_model_retrain() -> None:
    """Retrain ML models using the latest data."""
    from marketforge.utils.logger import setup_logging
    setup_logging()
    run_id = f"worker_retrain_{uuid.uuid4().hex[:8]}"
    log.info("worker.retrain.start", run_id=run_id)

    try:
        from marketforge.agents.ml_engineering.lead_agent import MLEngineerLeadAgent
        lead = MLEngineerLeadAgent()
        asyncio.run(lead.execute({}, None))
        log.info("worker.retrain.done", run_id=run_id)
    except Exception as exc:
        log.error("worker.retrain.failed", run_id=run_id, error=str(exc))
        raise


def job_cache_refresh() -> None:
    """Refresh the Redis dashboard cache."""
    from marketforge.memory.redis_cache import DashboardCache
    try:
        DashboardCache().invalidate()
        log.info("worker.cache_refresh.done")
    except Exception as exc:
        log.warning("worker.cache_refresh.failed", error=str(exc))


# ── CLI ────────────────────────────────────────────────────────────────────────

JOBS = {
    "ingest":   job_ingest,
    "analysis": job_weekly_analysis,
    "retrain":  job_model_retrain,
    "cache":    job_cache_refresh,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketForge pipeline worker")
    parser.add_argument("--run-now", choices=list(JOBS.keys()),
                        help="Run a single job immediately and exit")
    args = parser.parse_args()

    if args.run_now:
        log.info("worker.manual_run", job=args.run_now)
        JOBS[args.run_now]()
        return

    # ── Scheduled mode ────────────────────────────────────────────────────────
    scheduler = BlockingScheduler(timezone="UTC")

    # dag_ingest_primary — Tuesday + Thursday 07:00 UTC
    scheduler.add_job(job_ingest, CronTrigger.from_crontab("0 7 * * 2,4"),
                      id="ingest_primary", max_instances=1, coalesce=True)

    # dag_weekly_analysis — Monday 07:00 UTC
    scheduler.add_job(job_weekly_analysis, CronTrigger.from_crontab("0 7 * * 1"),
                      id="weekly_analysis", max_instances=1, coalesce=True)

    # dag_model_retrain — Sunday 02:00 UTC
    scheduler.add_job(job_model_retrain, CronTrigger.from_crontab("0 2 * * 0"),
                      id="model_retrain", max_instances=1, coalesce=True)

    # dag_dashboard_refresh — every 6 hours
    scheduler.add_job(job_cache_refresh, CronTrigger.from_crontab("0 */6 * * *"),
                      id="cache_refresh", max_instances=1, coalesce=True)

    log.info("worker.scheduler.started", jobs=[j.id for j in scheduler.get_jobs()])
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("worker.scheduler.stopped")


if __name__ == "__main__":
    import os
    run_now_env = os.getenv("RUN_NOW_ON_START")
    if run_now_env and run_now_env in JOBS:
        log.info("worker.env_trigger", job=run_now_env)
        JOBS[run_now_env]()
        # Clear the var hint so next restart goes to scheduler mode
    main()
