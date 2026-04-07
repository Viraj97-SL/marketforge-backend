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
        from marketforge.nlp.taxonomy import extract_skills_flat, classify_role
        from marketforge.memory.postgres import get_sync_engine, JobStore

        # Role → implied skills used when descriptions are too short for keyword match.
        # These are inserted at confidence=0.6 with method='role_inference' so
        # downstream analytics can weight them differently from extracted skills.
        _ROLE_IMPLIED: dict[str, list[tuple[str, str]]] = {
            "ml_engineer":              [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("scikit-learn", "ml_library"), ("SQL", "language"),
                                         ("Docker", "infra"), ("NumPy", "data_analysis")],
            "ai_engineer":              [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("LangChain", "llm_framework"), ("Docker", "infra"),
                                         ("REST API", "backend"), ("SQL", "language")],
            "data_scientist":           [("Python", "language"), ("scikit-learn", "ml_library"),
                                         ("SQL", "language"), ("Pandas", "data_analysis"),
                                         ("NumPy", "data_analysis"), ("Matplotlib", "visualisation")],
            "mlops_engineer":           [("Python", "language"), ("Kubernetes", "infra"),
                                         ("Docker", "infra"), ("MLflow", "mlops"),
                                         ("CI/CD", "devops"), ("Terraform", "infra")],
            "nlp_engineer":             [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("Hugging Face", "llm_framework"), ("BERT", "nlp"),
                                         ("spaCy", "nlp"), ("SQL", "language")],
            "computer_vision_engineer": [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("OpenCV", "computer_vision"), ("NumPy", "data_analysis"),
                                         ("Docker", "infra")],
            "research_scientist":       [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("JAX", "dl_framework"), ("NumPy", "data_analysis"),
                                         ("SQL", "language")],
            "applied_scientist":        [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("scikit-learn", "ml_library"), ("SQL", "language"),
                                         ("Pandas", "data_analysis")],
            "data_engineer":            [("Python", "language"), ("SQL", "language"),
                                         ("Apache Spark", "data_engineering"), ("dbt", "data_engineering"),
                                         ("Docker", "infra"), ("Airflow", "mlops")],
            "ai_safety_researcher":     [("Python", "language"), ("PyTorch", "dl_framework"),
                                         ("RLHF", "llm_technique"), ("NumPy", "data_analysis")],
            "ai_product_manager":       [("Python", "language"), ("SQL", "language")],
            "other":                    [("Python", "language"), ("SQL", "language")],
        }

        engine   = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        jobs_t   = "jobs"      if is_sqlite else "market.jobs"
        skills_t = "job_skills" if is_sqlite else "market.job_skills"

        # Fetch jobs with no skills yet AND also jobs with NULL role_category
        # (role classification runs for all un-classified jobs in the same pass)
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT j.job_id, j.title, j.description, j.role_category
                FROM {jobs_t} j
                WHERE NOT EXISTS (
                    SELECT 1 FROM {skills_t} s WHERE s.job_id = j.job_id
                )
            """)).fetchall()

        log.info("worker.ingest.nlp_start", jobs=len(rows))
        nlp_stats = {"gate1": 0, "gate2": 0, "gate3": 0, "role_inference": 0}
        job_store = JobStore()

        for row in rows:
            try:
                job_id, title, description, stored_role = row
                desc = description or ""
                text_blob = f"{title}\n{desc}"

                # ── Gate 1-3: taxonomy + spaCy + LLM ─────────────────────────
                skills = extract_skills_flat(text_blob)

                # ── Role classification ───────────────────────────────────────
                # classify_role uses title patterns; update DB if not yet set
                role_cat, exp_level = classify_role(title)
                if not stored_role:
                    with engine.connect() as conn:
                        if is_sqlite:
                            conn.execute(text(f"""
                                UPDATE {jobs_t}
                                SET role_category = :rc, experience_level = :el
                                WHERE job_id = :jid
                            """), {"rc": role_cat, "el": exp_level, "jid": job_id})
                        else:
                            conn.execute(text(f"""
                                UPDATE {jobs_t}
                                SET role_category = :rc, experience_level = :el
                                WHERE job_id = :jid
                                  AND (role_category IS NULL OR experience_level IS NULL)
                            """), {"rc": role_cat, "el": exp_level, "jid": job_id})
                        conn.commit()

                # ── Fallback: role-implied skills when description is too short ─
                # Short snippets (<150 chars) won't have tech keywords.
                # Infer likely skills from the classified role at lower confidence.
                desc_too_short = len(desc.strip()) < 150
                if desc_too_short or not skills:
                    implied_pairs = _ROLE_IMPLIED.get(role_cat, _ROLE_IMPLIED["other"])
                    # Keep only skills NOT already found by taxonomy gates
                    found_canonicals = {s[0] for s in skills}
                    implied_skills = [
                        (skill, cat, "role_inference", 0.6)
                        for skill, cat in implied_pairs
                        if skill not in found_canonicals
                    ]
                    skills = skills + implied_skills

                if skills:
                    job_store.upsert_skills(job_id, skills)
                    for _, _, gate, _ in skills:
                        nlp_stats[gate] = nlp_stats.get(gate, 0) + 1

            except Exception as exc:
                log.warning("worker.ingest.nlp_job_error", job_id=row[0], error=str(exc))

        log.info("worker.ingest.nlp_done", **nlp_stats)

        # ── Market analysis ───────────────────────────────────────────────────
        from marketforge.agents.market_analysis.lead_agent import MarketAnalystLeadAgent
        analyst = MarketAnalystLeadAgent()
        async def _run_analysis():
            plan = await analyst.plan({}, {})
            return await analyst.execute(plan, {})
        asyncio.run(_run_analysis())
        log.info("worker.ingest.analysis_done")

        # ── Cache invalidation ────────────────────────────────────────────────
        try:
            DashboardCache().invalidate()
            log.info("worker.ingest.cache_invalidated")
        except Exception as cache_exc:
            log.warning("worker.ingest.cache_skip", error=str(cache_exc))

        run_store.finish(run_id, status="success",
                         jobs_scraped=summary.get("jobs_new", 0),
                         jobs_new=summary.get("jobs_new", 0),
                         llm_cost_usd=cost_tracker.total_usd)
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
        async def _run_analysis():
            plan = await analyst.plan({}, {})
            return await analyst.execute(plan, {})
        asyncio.run(_run_analysis())

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
        async def _run_retrain():
            plan = await lead.plan({}, {})
            return await lead.execute(plan, {})
        asyncio.run(_run_retrain())
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
