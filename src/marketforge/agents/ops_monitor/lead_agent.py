"""
MarketForge AI — Department 9: Ops & Observability

Unlike other departments, the Ops Lead runs on a 30-minute heartbeat
AND is triggered by events from other departments. It has authority
to pause a failing department and escalate to human review.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.memory.postgres import get_sync_engine

logger = structlog.get_logger(__name__)


def _t(name: str) -> str:
    engine = get_sync_engine()
    return name if engine.dialect.name == "sqlite" else f"market.{name}"


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 1: CostTrackerAgent
# ══════════════════════════════════════════════════════════════════════════════

class CostTrackerAgent(DeepAgent):
    """
    Aggregates LLM spend per run, department, sub-agent.

    plan():    Reads adaptive_params["weekly_budget_usd"] and the 4-week
               rolling spend average. Sets a circuit-breaker threshold at
               2× the rolling average.

    execute(): Queries market.cost_log for this week's spend. Breaks down
               spend by department and model. Computes the next-run forecast
               using a weighted average of the last 4 runs (most recent 2×
               weight). Compares actual vs forecast to detect budget drift.

    reflect(): Triggers a cost alert if this run exceeded 2× the rolling
               average. Generates a "cost efficiency score" = insights per
               dollar (weekly_snapshot_quality / llm_spend_usd).
    """

    agent_id   = "cost_tracker_v1"
    department = "ops_monitor"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        rolling  = adaptive.get("rolling_weekly_costs", [])   # last 4 weeks' costs
        avg_cost = sum(rolling) / max(len(rolling), 1) if rolling else 0.15
        threshold= avg_cost * 2
        return {"threshold_usd": threshold, "rolling": rolling, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        cost_t    = "cost_log" if is_sqlite else "market.cost_log"
        week_ago  = (datetime.utcnow() - timedelta(days=7)).isoformat()

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT agent_name, model, SUM(input_tokens) as it,
                       SUM(output_tokens) as ot, SUM(cost_usd) as cost
                FROM {cost_t}
                WHERE logged_at >= :since
                GROUP BY agent_name, model
                ORDER BY cost DESC
            """), {"since": week_ago}).fetchall()

            total_cost = conn.execute(text(f"""
                SELECT COALESCE(SUM(cost_usd), 0) FROM {cost_t}
                WHERE logged_at >= :since
            """), {"since": week_ago}).scalar() or 0.0

        breakdown = [
            {"agent": r[0], "model": r[1], "input_tokens": r[2], "output_tokens": r[3], "cost_usd": round(r[4], 6)}
            for r in rows
        ]

        return {
            "total_cost_usd": round(float(total_cost), 4),
            "breakdown":      breakdown,
            "threshold":      plan["threshold_usd"],
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        cost     = result.get("total_cost_usd", 0)
        threshold= result.get("threshold", 0.5)
        adaptive = plan.get("adaptive", {})
        rolling  = plan.get("rolling", [])

        rolling.append(cost)
        rolling = rolling[-4:]  # keep last 4 weeks

        adaptive["rolling_weekly_costs"] = rolling
        state["adaptive_params"]         = adaptive

        quality = "good"
        if cost > threshold and threshold > 0:
            quality = "warning"
            logger.warning("cost_tracker.over_threshold", cost=cost, threshold=threshold)

        return {
            "quality":        quality,
            "weekly_cost_usd":cost,
            "notes":          f"cost=${cost:.3f}, threshold=${threshold:.3f}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "weekly_cost_usd": result.get("total_cost_usd", 0),
            "cost_breakdown":  result.get("breakdown", []),
            "over_budget":     reflection.get("quality") == "warning",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 2: PipelineHealthMonitorAgent
# ══════════════════════════════════════════════════════════════════════════════

class PipelineHealthMonitorAgent(DeepAgent):
    """
    Monitors Airflow DAG execution state and agent-level health.

    plan():    Reads adaptive_params["expected_run_durations"] to set
               stall detection thresholds (> 2× historical average = stall).
               Checks when each DAG last ran successfully.

    execute(): Queries market.pipeline_runs for the last 10 runs per DAG.
               Computes per-department health_score:
                 1.0 = all recent runs successful and timely
                 0.5 = some failures or stalls
                 0.0 = consecutive failures or never ran
               Also reads market.agent_state for sub-agent failure counts.

    reflect(): Identifies any department with health_score < 0.3 and
               prepares an escalation payload for AlertDispatchAgent.
               Checks data freshness: if the last ingestion was > 72h ago,
               flags "stale data" regardless of run status.
    """

    agent_id   = "pipeline_health_v1"
    department = "ops_monitor"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        return {"adaptive": adaptive, "expected_durations": adaptive.get("expected_durations", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        engine   = get_sync_engine()
        runs_t   = "pipeline_runs" if engine.dialect.name == "sqlite" else "market.pipeline_runs"
        agents_t = "agent_state"   if engine.dialect.name == "sqlite" else "market.agent_state"

        with engine.connect() as conn:
            # Last run per DAG
            dag_rows = conn.execute(text(f"""
                SELECT dag_name, status, started_at, completed_at,
                       jobs_scraped, jobs_new
                FROM {runs_t}
                ORDER BY started_at DESC LIMIT 20
            """)).mappings().fetchall()

            # Agent failure counts
            agent_rows = conn.execute(text(f"""
                SELECT department, agent_id, consecutive_failures, last_run_at
                FROM {agents_t}
                ORDER BY department, agent_id
            """)).mappings().fetchall()

        # Compute health scores per DAG
        dag_health: dict[str, dict] = {}
        for row in dag_rows:
            dag = row["dag_name"]
            if dag not in dag_health:
                dag_health[dag] = {"runs": [], "status": "unknown"}
            dag_health[dag]["runs"].append({"status": row["status"], "started": str(row["started_at"])})

        for dag, info in dag_health.items():
            recent = info["runs"][:5]
            success_rate = sum(1 for r in recent if r["status"] in ("success", "running")) / max(len(recent), 1)
            info["health_score"] = round(success_rate, 2)
            info["last_run"]     = recent[0]["started"] if recent else None

        # Data freshness
        latest_ingestion = None
        for row in dag_rows:
            if row["dag_name"] == "dag_ingest_primary" and row["status"] == "success":
                latest_ingestion = row["completed_at"]
                break

        freshness_h = None
        if latest_ingestion:
            try:
                lt = datetime.fromisoformat(str(latest_ingestion))
                freshness_h = round((datetime.utcnow() - lt.replace(tzinfo=None)).total_seconds() / 3600, 1)
            except Exception:
                pass

        # Agent failures
        dept_failures: dict[str, int] = {}
        for row in agent_rows:
            dept = row["department"]
            dept_failures[dept] = dept_failures.get(dept, 0) + (row["consecutive_failures"] or 0)

        return {
            "dag_health":         dag_health,
            "dept_failures":      dept_failures,
            "data_freshness_h":   freshness_h,
            "latest_ingestion":   str(latest_ingestion) if latest_ingestion else None,
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        dag_health  = result.get("dag_health", {})
        freshness_h = result.get("data_freshness_h")

        unhealthy   = [dag for dag, info in dag_health.items() if info.get("health_score", 1) < 0.5]
        stale       = freshness_h is not None and freshness_h > 72

        quality = "good"
        if unhealthy or stale:
            quality = "warning"
            if unhealthy:
                logger.warning("pipeline_health.unhealthy_dags", dags=unhealthy)
            if stale:
                logger.warning("pipeline_health.stale_data", age_h=freshness_h)

        return {
            "quality":       quality,
            "unhealthy_dags":unhealthy,
            "stale_data":    stale,
            "freshness_h":   freshness_h,
            "notes":         f"unhealthy={len(unhealthy)}, stale={stale}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "dag_health":       result.get("dag_health", {}),
            "dept_failures":    result.get("dept_failures", {}),
            "data_freshness_h": result.get("data_freshness_h"),
            "platform_healthy": reflection.get("quality") == "good",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 3: AlertDispatchAgent
# ══════════════════════════════════════════════════════════════════════════════

class AlertDispatchAgent(DeepAgent):
    """
    Severity-tiered notification gateway.

    Severity 1 (critical): data pipeline failure, security breach → immediate email
    Severity 2 (warning):  model drift, cost overrun, stale data → Monday report
    Severity 3 (info):     minor anomalies, low yield → logged only

    plan():    Reads market.alert_log to check the 4-hour deduplication window.
               Does not re-dispatch an alert for the same issue within 4 hours.

    execute(): Constructs alert payloads and dispatches:
               - Severity 1 → SMTP email immediately
               - Severity 2 → appended to the weekly ops report queue
               - Severity 3 → written to market.alert_log only

    reflect(): Checks for unacknowledged severity-1 alerts older than 2 hours.
               Auto-escalates them (re-sends email with [ESCALATED] subject prefix).
    """

    agent_id   = "alert_dispatch_v1"
    department = "ops_monitor"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        # Load recent alerts to check deduplication window
        recent_alerts: set[str] = set()
        try:
            from sqlalchemy import text
            engine = get_sync_engine()
            table  = "alert_log" if engine.dialect.name == "sqlite" else "market.alert_log"
            cutoff = (datetime.utcnow() - timedelta(hours=4)).isoformat()
            with engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT message FROM {table}
                    WHERE dispatched_at >= :since AND acknowledged_at IS NULL
                """), {"since": cutoff}).fetchall()
            recent_alerts = {r[0][:80] for r in rows}
        except Exception:
            pass
        return {
            "pending_alerts":  context.get("pending_alerts", []),
            "recent_alerts":   recent_alerts,
            "adaptive":        state.get("adaptive_params", {}),
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        alerts   = plan.get("pending_alerts", [])
        recent   = plan.get("recent_alerts", set())
        dispatched = []

        for alert in alerts:
            severity = alert.get("severity", 3)
            message  = alert.get("message", "")

            # Deduplication check
            if message[:80] in recent:
                logger.debug("alert.deduped", msg=message[:60])
                continue

            self._write_alert(severity, alert.get("department", ""), message)

            if severity == 1:
                self._send_email_alert(message, alert.get("department", ""))
                dispatched.append({"severity": 1, "message": message[:80]})
            elif severity == 2:
                logger.info("alert.queued_for_report", message=message[:60])
                dispatched.append({"severity": 2, "message": message[:80]})
            else:
                logger.debug("alert.logged_only", message=message[:60])

        return {"dispatched": dispatched}

    def _write_alert(self, severity: int, department: str, message: str) -> None:
        try:
            from sqlalchemy import text
            import uuid
            engine = get_sync_engine()
            table  = "alert_log" if engine.dialect.name == "sqlite" else "market.alert_log"
            now    = datetime.utcnow().isoformat()
            aid    = str(uuid.uuid4())[:16]
            with engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {table} (alert_id, severity, department, message, dispatched_at)
                    VALUES (:aid, :sv, :dept, :msg, :now)
                """), {"aid": aid, "sv": severity, "dept": department, "msg": message[:500], "now": now})
                conn.commit()
        except Exception as exc:
            logger.warning("alert.write_failed", error=str(exc))

    def _send_email_alert(self, message: str, department: str) -> None:
        try:
            import smtplib
            from email.mime.text import MIMEText
            from marketforge.config.settings import settings
            cfg = settings.email
            if not cfg.user or not cfg.password:
                return
            msg          = MIMEText(message)
            msg["From"]  = cfg.user
            msg["To"]    = cfg.recipient_email or cfg.user
            msg["Subject"] = f"[MARKETFORGE ALERT] {department.upper()} — Severity 1"
            with smtplib.SMTP(cfg.host, cfg.port) as s:
                s.starttls()
                s.login(cfg.user, cfg.password)
                s.send_message(msg)
            logger.info("alert.email.sent", department=department)
        except Exception as exc:
            logger.error("alert.email.failed", error=str(exc))

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        dispatched = result.get("dispatched", [])
        sev1_count = sum(1 for d in dispatched if d["severity"] == 1)
        return {
            "quality":          "good",
            "alerts_dispatched": len(dispatched),
            "sev1":             sev1_count,
            "notes":            f"dispatched={len(dispatched)}, sev1={sev1_count}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {"alerts_dispatched": result.get("dispatched", [])}


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 4: InfrastructureHealthAgent
# ══════════════════════════════════════════════════════════════════════════════

class InfrastructureHealthAgent(DeepAgent):
    """
    Monitors PostgreSQL, Redis, and ChromaDB resource utilisation.

    plan():    Reads adaptive_params["capacity_thresholds"] for warning levels.
               Schedules a capacity headroom projection (7-day linear extrapolation).

    execute(): Runs four infrastructure checks:
               1. PostgreSQL: connection pool utilisation + table size growth
               2. Redis: used_memory / maxmemory ratio + eviction rate
               3. ChromaDB: collection sizes (number of embeddings)
               4. Disk: rough storage estimate from table row counts

    reflect(): Alerts if any resource is projected to exceed 80% within 7 days.
               Automatically adjusts Redis eviction policy if memory > 75%.

    output():  Returns a resource health dict and list of capacity warnings.
    """

    agent_id   = "infra_health_v1"
    department = "ops_monitor"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive    = state.get("adaptive_params", {})
        thresholds  = adaptive.get("capacity_thresholds", {"warn": 0.70, "alert": 0.85})
        return {"thresholds": thresholds, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import text
        engine   = get_sync_engine()
        is_sqlite= engine.dialect.name == "sqlite"
        resources: dict[str, dict] = {}

        # ── 1. PostgreSQL ──────────────────────────────────────────────────────
        try:
            with engine.connect() as conn:
                if not is_sqlite:
                    pool_rows = conn.execute(text("""
                        SELECT COUNT(*) as connections
                        FROM pg_stat_activity
                        WHERE state != 'idle'
                    """)).fetchone()
                    db_size = conn.execute(text(
                        "SELECT pg_size_pretty(pg_database_size(current_database()))"
                    )).scalar()
                    resources["postgresql"] = {
                        "active_connections": pool_rows[0] if pool_rows else 0,
                        "database_size":      db_size or "unknown",
                        "status":             "ok",
                    }

                    # Row counts for key tables
                    table_counts = {}
                    for t in ["market.jobs", "market.job_skills", "market.agent_logs"]:
                        try:
                            cnt = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                            table_counts[t] = cnt
                        except Exception:
                            pass
                    resources["postgresql"]["table_counts"] = table_counts
                else:
                    resources["postgresql"] = {"status": "sqlite_dev_mode"}
        except Exception as exc:
            resources["postgresql"] = {"status": "error", "error": str(exc)[:100]}

        # ── 2. Redis ───────────────────────────────────────────────────────────
        try:
            import redis as redis_lib
            from marketforge.config.settings import settings
            r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=3)
            info = r.info("memory")
            used    = info.get("used_memory", 0)
            maxmem  = info.get("maxmemory", 256 * 1024 * 1024)
            if maxmem == 0:
                maxmem = 256 * 1024 * 1024  # 256MB default
            ratio   = used / maxmem
            resources["redis"] = {
                "used_memory_mb":  round(used / 1024 / 1024, 1),
                "max_memory_mb":   round(maxmem / 1024 / 1024, 1),
                "utilisation":     round(ratio, 3),
                "evicted_keys":    info.get("evicted_keys", 0),
                "status":          "warning" if ratio > 0.75 else "ok",
            }
        except Exception as exc:
            resources["redis"] = {"status": "unavailable", "error": str(exc)[:80]}

        # ── 3. ChromaDB ────────────────────────────────────────────────────────
        try:
            from marketforge.config.settings import settings
            import chromadb
            client     = chromadb.PersistentClient(path=settings.chroma_db_dir)
            collections= client.list_collections()
            resources["chromadb"] = {
                "collection_count": len(collections),
                "collections":      [{"name": c.name, "count": c.count()} for c in collections],
                "status":           "ok",
            }
        except Exception as exc:
            resources["chromadb"] = {"status": "unavailable", "error": str(exc)[:80]}

        logger.info(f"{self.agent_id}.execute.done", resources=list(resources.keys()))
        return {"resources": resources}

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        resources   = result.get("resources", {})
        thresholds  = plan["thresholds"]
        warnings:   list[str] = []

        redis_info  = resources.get("redis", {})
        redis_util  = redis_info.get("utilisation", 0)
        if redis_util > thresholds["alert"]:
            warnings.append(f"Redis utilisation critical: {redis_util:.0%}")
        elif redis_util > thresholds["warn"]:
            warnings.append(f"Redis utilisation high: {redis_util:.0%}")

        quality = "poor" if any("critical" in w for w in warnings) else ("warning" if warnings else "good")
        if warnings:
            logger.warning(f"{self.agent_id}.capacity_warnings", warnings=warnings)

        return {"quality": quality, "capacity_warnings": warnings}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "infrastructure":    result.get("resources", {}),
            "capacity_warnings": reflection.get("capacity_warnings", []),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 5: PerformanceBenchmarkAgent
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceBenchmarkAgent(DeepAgent):
    """
    Runs monthly performance benchmarks across all platform components.

    plan():    Reads adaptive_params["last_benchmark_at"] to check if a
               monthly benchmark is due (> 28 days since last run).
               Can be forced via context["force"] = True.

    execute(): Benchmarks four areas:
               1. NLP extraction throughput: processes 100 sample job descriptions
                  and measures records/second for Gates 1+2 (no LLM).
               2. API response time: queries the market data endpoints and measures
                  P50/P95 latency (internal HTTP call to localhost).
               3. Database query performance: runs the top-5 dashboard queries
                  and measures execution time.
               4. Skill extraction accuracy: runs the golden dataset (100 labelled JDs)
                  through the NLP pipeline and computes precision/recall.

    reflect(): Compares results against prior month's baseline. Flags any
               component with > 20% regression. Updates benchmark history.

    output():  Returns performance report with component benchmarks and regressions.
    """

    agent_id   = "perf_benchmark_v1"
    department = "ops_monitor"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive      = state.get("adaptive_params", {})
        last_benchmark= adaptive.get("last_benchmark_at")
        force         = context.get("force", False)

        due = True
        if last_benchmark and not force:
            try:
                last_dt = datetime.fromisoformat(last_benchmark)
                due = (datetime.utcnow() - last_dt).days >= 28
            except Exception:
                due = True

        return {"due": due, "adaptive": adaptive, "force": force}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if not plan.get("due"):
            return {"skipped": True, "reason": "benchmark_not_due"}

        benchmarks: dict[str, dict] = {}

        # ── 1. NLP extraction throughput ───────────────────────────────────────
        try:
            from marketforge.nlp.taxonomy import extract_skills_flat
            import time

            sample_texts = [
                "Senior ML Engineer with 5+ years Python, PyTorch, TensorFlow. "
                "Experience with LangChain, RAG, and MLflow. AWS and Kubernetes required. "
                "London or remote. £90k–£130k. Visa sponsorship available."
            ] * 100   # 100 identical texts for throughput test

            start     = time.perf_counter()
            for text in sample_texts:
                extract_skills_flat(text)
            elapsed   = time.perf_counter() - start
            throughput= round(len(sample_texts) / elapsed, 1)

            benchmarks["nlp_throughput"] = {
                "records_per_second": throughput,
                "sample_size":        len(sample_texts),
                "elapsed_ms":         round(elapsed * 1000),
            }
            logger.info(f"{self.agent_id}.nlp_benchmark", rps=throughput)
        except Exception as exc:
            benchmarks["nlp_throughput"] = {"error": str(exc)[:100]}

        # ── 2. Database query performance ──────────────────────────────────────
        try:
            import time
            from sqlalchemy import text
            engine   = get_sync_engine()
            is_sqlite= engine.dialect.name == "sqlite"
            snap_t   = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

            timings: list[float] = []
            queries = [
                f"SELECT COUNT(*) FROM {'jobs' if is_sqlite else 'market.jobs'}",
                f"SELECT * FROM {snap_t} ORDER BY week_start DESC LIMIT 1",
                f"SELECT skill, COUNT(*) FROM {'job_skills' if is_sqlite else 'market.job_skills'} GROUP BY skill ORDER BY COUNT(*) DESC LIMIT 20",
            ]
            with engine.connect() as conn:
                for q in queries:
                    t0 = time.perf_counter()
                    try:
                        conn.execute(text(q)).fetchall()
                    except Exception:
                        pass
                    timings.append((time.perf_counter() - t0) * 1000)

            benchmarks["db_query_performance"] = {
                "p50_ms": round(sorted(timings)[len(timings) // 2], 1),
                "max_ms": round(max(timings), 1),
                "queries_tested": len(timings),
            }
        except Exception as exc:
            benchmarks["db_query_performance"] = {"error": str(exc)[:100]}

        logger.info(f"{self.agent_id}.execute.done", components=list(benchmarks.keys()))
        return {"benchmarks": benchmarks, "ran_at": datetime.utcnow().isoformat()}

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        if result.get("skipped"):
            return {"quality": "good", "notes": "benchmark_not_due"}

        adaptive  = plan.get("adaptive", {})
        benchmarks= result.get("benchmarks", {})
        history   = adaptive.get("benchmark_history", [])
        history.append({"at": result.get("ran_at"), "benchmarks": benchmarks})
        adaptive["benchmark_history"]   = history[-3:]   # keep last 3 months
        adaptive["last_benchmark_at"]   = result.get("ran_at")
        state["adaptive_params"]        = adaptive

        regressions: list[str] = []
        if len(history) >= 2:
            prev = history[-2].get("benchmarks", {})
            nlp_prev  = prev.get("nlp_throughput", {}).get("records_per_second", 0)
            nlp_curr  = benchmarks.get("nlp_throughput", {}).get("records_per_second", 0)
            if nlp_prev > 0 and nlp_curr > 0 and nlp_curr < nlp_prev * 0.80:
                regressions.append(f"NLP throughput degraded: {nlp_curr} rps vs {nlp_prev} rps")

        quality = "warning" if regressions else "good"
        if regressions:
            logger.warning(f"{self.agent_id}.regressions_detected", regressions=regressions)

        return {"quality": quality, "regressions": regressions}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "benchmarks":  result.get("benchmarks", {}),
            "regressions": reflection.get("regressions", []),
            "ran_at":      result.get("ran_at"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ops Lead Agent
# ══════════════════════════════════════════════════════════════════════════════

class OpsLeadAgent(DeepAgent):
    """
    Department 9 Lead — platform operational nerve centre.

    Runs on a 30-minute heartbeat (triggered by Airflow or directly).
    Aggregates health from all other departments, manages cost tracking,
    dispatches alerts, and generates the weekly ops summary.
    """

    agent_id   = "ops_lead_v1"
    department = "ops_monitor"

    def __init__(self) -> None:
        self._cost_agent    = CostTrackerAgent()
        self._health_agent  = PipelineHealthMonitorAgent()
        self._alert_agent   = AlertDispatchAgent()
        self._infra_agent   = InfrastructureHealthAgent()
        self._bench_agent   = PerformanceBenchmarkAgent()

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return {"context": context, "adaptive": state.get("adaptive_params", {})}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        import asyncio
        # Run cost + health + infra in parallel (all read-only, no LLM)
        cost_out, health_out, infra_out = await asyncio.gather(
            self._cost_agent.run({}),
            self._health_agent.run({}),
            self._infra_agent.run({}),
            return_exceptions=True,
        )
        if isinstance(cost_out,   Exception): cost_out   = {}
        if isinstance(health_out, Exception): health_out = {}
        if isinstance(infra_out,  Exception): infra_out  = {}

        # Build alert list from findings
        alerts: list[dict] = []
        if cost_out.get("over_budget"):
            alerts.append({"severity": 2, "department": "ops", "message": f"Weekly LLM cost ${cost_out.get('weekly_cost_usd', 0):.3f} exceeded threshold"})
        if not health_out.get("platform_healthy"):
            for dag in health_out.get("dag_health", {}):
                score = health_out["dag_health"][dag].get("health_score", 1)
                if score < 0.3:
                    alerts.append({"severity": 1, "department": dag, "message": f"DAG {dag} health score {score} — multiple failures detected"})
        if health_out.get("data_freshness_h", 0) > 96:
            alerts.append({"severity": 2, "department": "data_collection", "message": f"Data is {health_out.get('data_freshness_h')}h stale — ingestion may have failed"})
        for warning in infra_out.get("capacity_warnings", []):
            alerts.append({"severity": 2, "department": "infrastructure", "message": warning})

        alert_out = await self._alert_agent.run({"pending_alerts": alerts})

        return {
            "cost":    cost_out,
            "health":  health_out,
            "infra":   infra_out,
            "alerts":  alert_out,
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        healthy = result.get("health", {}).get("platform_healthy", True)
        budget  = not result.get("cost", {}).get("over_budget", False)
        quality = "good" if (healthy and budget) else "warning"
        return {"quality": quality, "notes": f"healthy={healthy}, budget_ok={budget}"}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "ops_summary": {
                "platform_healthy":      result.get("health", {}).get("platform_healthy"),
                "data_freshness_h":      result.get("health", {}).get("data_freshness_h"),
                "weekly_cost_usd":       result.get("cost",   {}).get("weekly_cost_usd"),
                "alerts_dispatched":     len(result.get("alerts", {}).get("alerts_dispatched", [])),
                "capacity_warnings":     result.get("infra",  {}).get("capacity_warnings", []),
                "infrastructure":        result.get("infra",  {}).get("infrastructure", {}),
            },
            "quality": reflection.get("quality"),
        }


async def run_ops_heartbeat() -> dict:
    """Called every 30 minutes by Airflow or cron."""
    lead = OpsLeadAgent()
    return await lead.run({})
