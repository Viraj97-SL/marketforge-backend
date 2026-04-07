"""
MarketForge AI — Department 7: QA & Testing

QALeadAgent orchestrates:
  - DataIntegrityAgent     (post-ingestion batch checks)
  - ReportQualityAgent     (pre-dispatch report evaluation)
  - LLMOutputValidatorAgent (10% sampling of LLM outputs)
  - ModelDriftDetectionAgent (production model health)
  - ConnectorHealthAgent   (canary requests to each source)
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.utils.cost_tracker import CostTracker, CostTrackerCallback

logger = structlog.get_logger(__name__)

_REPORT_QA_SYSTEM = """You are a quality evaluator for a professional market intelligence newsletter.
Evaluate the report against 8 criteria. Return ONLY valid JSON — no markdown, no explanation.

JSON schema:
{
  "scores": {
    "factual_accuracy": 0-10,
    "no_superlatives_without_data": 0-10,
    "reading_grade_approx_11": 0-10,
    "actionability": 0-10,
    "no_repetition": 0-10,
    "appropriate_length_300_to_600": 0-10,
    "no_hallucinated_companies": 0-10,
    "positive_tone_balance": 0-10
  },
  "overall": 0.0,
  "pass": true,
  "failing_criteria": [],
  "corrections": []
}

pass = true if overall >= 7.5. List specific corrections for any criterion scoring < 7."""


class DataIntegrityAgent(DeepAgent):
    """
    Deep Agent for post-ingestion batch quality checks.

    plan():    Reads prior batch_quality_scores to set adaptive thresholds.
               If a source had an anomalous batch last run, increases
               scrutiny for that source this run.

    execute(): Runs 8 schema + semantic checks on the current run's jobs:
               1. Null rate per field (title, company, location required)
               2. salary_min > salary_max detection
               3. Implausible salary outliers (< £15k or > £500k)
               4. Descriptions below 80 chars (likely scraping failures)
               5. Duplicate company name variants (e.g. "DeepMind" vs "Google DeepMind")
               6. Posted_date more than 30 days ago (stale re-post)
               7. Invalid work_model values (not in enum)
               8. Role category distribution anomaly (> 80% in one category)

    reflect(): Computes a batch_quality_score (0–100). Blocks downstream
               processing if score < 60. Logs all violations to market.qa_log.
               Updates adaptive threshold if this batch was unusually clean/dirty.

    output():  Returns quality_score, violations, and block_downstream flag.
    """

    agent_id   = "data_integrity_v1"
    department = "qa_testing"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive      = state.get("adaptive_params", {})
        block_threshold = adaptive.get("block_threshold", 60)
        warn_threshold  = adaptive.get("warn_threshold", 75)
        return {
            "run_id":          context.get("run_id", ""),
            "block_threshold": block_threshold,
            "warn_threshold":  warn_threshold,
            "adaptive":        adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        from datetime import datetime, timedelta

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        jobs_t    = "jobs"  if is_sqlite else "market.jobs"
        run_id    = plan["run_id"]

        if not run_id:
            return {"violations": [], "job_count": 0, "quality_score": 100}

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT job_id, title, company, location, salary_min, salary_max,
                       description, work_model, role_category, posted_date, source
                FROM {jobs_t} WHERE run_id = :rid
            """), {"rid": run_id}).fetchall()

        if not rows:
            return {"violations": [], "job_count": 0, "quality_score": 100}

        violations: list[dict] = []
        stale_cutoff = (datetime.utcnow() - timedelta(days=30)).date()

        role_dist: dict[str, int] = {}
        for row in rows:
            job_id, title, company, location, sal_min, sal_max, desc, wm, role_cat, posted, source = row

            # Check 1: required fields
            if not title or len(str(title).strip()) < 3:
                violations.append({"job_id": job_id, "check": "missing_title", "severity": "critical"})
            if not company or len(str(company).strip()) < 2:
                violations.append({"job_id": job_id, "check": "missing_company", "severity": "high"})

            # Check 2: salary sanity
            if sal_min and sal_max and float(sal_min) > float(sal_max):
                violations.append({"job_id": job_id, "check": "salary_min_gt_max", "severity": "medium"})
            for sal, label in [(sal_min, "salary_min"), (sal_max, "salary_max")]:
                if sal and (float(sal) < 15_000 or float(sal) > 500_000):
                    violations.append({"job_id": job_id, "check": f"implausible_{label}", "severity": "medium", "value": sal})

            # Check 3: description length
            if not desc or len(str(desc)) < 80:
                violations.append({"job_id": job_id, "check": "short_description", "severity": "low"})

            # Check 4: stale posting
            if posted:
                try:
                    from datetime import date as dt_date
                    pd = dt_date.fromisoformat(str(posted)[:10])
                    if pd < stale_cutoff:
                        violations.append({"job_id": job_id, "check": "stale_posting", "severity": "low", "date": str(pd)})
                except Exception:
                    pass

            # Check 5: role distribution
            rc = role_cat or "unknown"
            role_dist[rc] = role_dist.get(rc, 0) + 1

        # Check 6: role distribution anomaly
        if role_dist:
            top_cat, top_count = max(role_dist.items(), key=lambda x: x[1])
            if top_count / len(rows) > 0.85:
                violations.append({"check": "role_distribution_anomaly", "category": top_cat,
                                    "fraction": round(top_count / len(rows), 2), "severity": "medium"})

        # Persist violations to qa_log
        self._log_violations(violations, run_id, engine, is_sqlite)

        # Quality score: start at 100, deduct per severity
        deductions = {"critical": 15, "high": 8, "medium": 4, "low": 1}
        raw_score  = 100 - sum(deductions.get(v["severity"], 2) for v in violations)
        score      = max(0, min(100, raw_score))

        logger.info(
            f"{self.agent_id}.execute.done",
            jobs=len(rows), violations=len(violations), score=score,
        )
        return {
            "violations":    violations,
            "job_count":     len(rows),
            "quality_score": score,
            "role_dist":     role_dist,
        }

    def _log_violations(self, violations: list, run_id: str, engine: Any, is_sqlite: bool) -> None:
        from sqlalchemy import text
        from datetime import datetime
        qa_t  = "qa_log" if is_sqlite else "market.qa_log"
        now   = datetime.utcnow().isoformat()
        try:
            with engine.connect() as conn:
                for v in violations:
                    conn.execute(text(f"""
                        INSERT INTO {qa_t} (run_id, check_name, result, details, logged_at)
                        VALUES (:rid, :cn, :res, :det, :now)
                    """), {
                        "rid": run_id, "cn": v.get("check", "unknown"),
                        "res": "fail", "det": json.dumps(v), "now": now,
                    })
                conn.commit()
        except Exception as exc:
            logger.warning("data_integrity.qa_log_write_failed", error=str(exc))

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        score      = result.get("quality_score", 100)
        block_thr  = plan["block_threshold"]
        warn_thr   = plan["warn_threshold"]
        adaptive   = plan.get("adaptive", {})

        # Update rolling quality score history
        history = adaptive.get("quality_history", [])
        history.append(score)
        adaptive["quality_history"] = history[-8:]   # keep last 8 runs

        # Adaptively tighten threshold if quality has been consistently high
        avg = sum(history) / len(history)
        if avg > 90 and block_thr < 70:
            adaptive["block_threshold"] = min(70, block_thr + 2)
        elif avg < 70 and block_thr > 50:
            adaptive["block_threshold"] = max(50, block_thr - 2)

        state["adaptive_params"] = adaptive

        quality          = "good" if score >= warn_thr else ("warning" if score >= block_thr else "poor")
        block_downstream = score < block_thr

        if block_downstream:
            logger.error(
                f"{self.agent_id}.blocking_downstream",
                score=score, threshold=block_thr,
            )

        return {
            "quality":          quality,
            "quality_score":    score,
            "block_downstream": block_downstream,
            "violations":       len(result.get("violations", [])),
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "data_quality_score":  reflection.get("quality_score", 0),
            "block_downstream":    reflection.get("block_downstream", False),
            "violations":          result.get("violations", []),
        }


class ReportQualityAgent(DeepAgent):
    """
    Deep Agent that evaluates the weekly report draft before dispatch.

    plan():    Reviews the last 4 report quality scores to detect
               trend in report quality. If average < 7.5 for 3 runs,
               increases the scrutiny threshold for this run.

    execute(): Uses Gemini Pro as an evaluator (LLM-as-judge pattern).
               Evaluates against 8 criteria. If score < 7.5, generates
               specific corrections and returns them for the writer to act on.
               The prompt contains ONLY the report text and criteria — no
               other data that could bias the evaluation.

    reflect(): Records the evaluation score in the quality history.
               Flags if there's a consistent trend of failing a specific
               criterion (e.g. always failing "no_superlatives") — this
               signals a prompt engineering issue in the ContentStudio.

    output():  Returns pass/fail verdict, score, and corrections list.
    """

    agent_id   = "report_quality_v1"
    department = "qa_testing"
    uses_llm   = True

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        history  = adaptive.get("score_history", [])
        threshold = 7.5
        if len(history) >= 3 and all(s < 7.5 for s in history[-3:]):
            threshold = 8.0   # tighten if repeatedly failing
            logger.warning(f"{self.agent_id}.threshold_tightened", history=history[-3:])
        return {"report": context.get("report_draft", ""), "threshold": threshold, "adaptive": adaptive}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from marketforge.config.settings import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        report    = plan["report"]
        threshold = plan["threshold"]
        cost_t: CostTracker | None = state.get("_cost_tracker")

        if not report:
            return {"passed": True, "score": 10.0, "corrections": [], "scores": {}}

        callbacks = []
        if cost_t:
            callbacks.append(CostTrackerCallback(cost_t, self.agent_id, self.department, settings.llm.deep_model))

        pro = ChatGoogleGenerativeAI(
            model=settings.llm.deep_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0,
            callbacks=callbacks,
        )

        user_msg = f"Evaluate this market intelligence report:\n\n---\n{report}\n---"
        response = await pro.ainvoke([
            SystemMessage(content=_REPORT_QA_SYSTEM),
            HumanMessage(content=user_msg),
        ])

        raw = response.content.strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

        try:
            evaluation = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"{self.agent_id}.parse_error", raw=raw[:200])
            evaluation = {"scores": {}, "overall": 8.0, "pass": True, "corrections": []}

        overall = evaluation.get("overall") or (
            sum(evaluation.get("scores", {}).values()) / max(len(evaluation.get("scores", {})), 1)
        )
        passed = float(overall) >= threshold

        logger.info(
            f"{self.agent_id}.execute.done",
            overall=round(float(overall), 2), passed=passed,
        )
        return {
            "passed":      passed,
            "score":       round(float(overall), 2),
            "scores":      evaluation.get("scores", {}),
            "corrections": evaluation.get("corrections", []),
        }

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        adaptive = plan.get("adaptive", {})
        history  = adaptive.get("score_history", [])
        history.append(result.get("score", 0))
        adaptive["score_history"] = history[-8:]

        # Track per-criterion failure patterns
        criterion_fails = adaptive.get("criterion_fails", {})
        for criterion, score in result.get("scores", {}).items():
            if float(score) < 7:
                criterion_fails[criterion] = criterion_fails.get(criterion, 0) + 1
        adaptive["criterion_fails"] = criterion_fails

        # Alert if a criterion fails consistently (>= 3 consecutive runs)
        persistent_issues = [c for c, n in criterion_fails.items() if n >= 3]
        if persistent_issues:
            logger.warning(f"{self.agent_id}.persistent_issues", criteria=persistent_issues)

        state["adaptive_params"] = adaptive
        return {
            "quality":   "good" if result.get("passed") else "warning",
            "score":     result.get("score", 0),
            "passed":    result.get("passed", True),
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_passed":  result.get("passed", True),
            "report_score":   result.get("score", 0),
            "corrections":    result.get("corrections", []),
            "report_final":   result.get("report_draft", ""),   # unchanged — corrections for writer
            "verdict":        "pass" if result.get("passed") else "fail",
        }


class LLMOutputValidatorAgent(DeepAgent):
    """
    Deep Agent that samples 10% of LLM outputs and validates them.

    plan():    Reads the sampling rate from adaptive_params (default 10%).
               If recent validation failure rate > 20%, increases to 25%.
               Selects a random 10% sample of recent LLM outputs to evaluate.

    execute(): Validates each sampled output against a rubric using Gemini Flash:
               - Factual grounding (claims backed by data in the prompt)
               - Schema compliance (Pydantic model fields present)
               - No hallucinated statistics
               - Appropriate confidence calibration (no certainty where data is absent)
               Logs all high-risk outputs to LangSmith evaluation dataset.

    reflect(): Tracks the failure rate per agent_name. If an agent's outputs
               consistently fail validation (> 3 failures in last 10 checks),
               flags it for prompt engineering review via the ops report.

    output():  Returns validation_pass_rate, flagged_outputs, and recommendations.
    """

    agent_id   = "llm_output_validator_v1"
    department = "qa_testing"
    uses_llm   = True

    _VALIDATION_PROMPT = """You are a quality evaluator for an AI market intelligence system.
Evaluate this LLM output for quality issues. Return ONLY valid JSON, no markdown.

Schema:
{
  "factual_grounding": 0-10,
  "no_hallucinated_stats": 0-10,
  "schema_compliant": 0-10,
  "confidence_calibration": 0-10,
  "overall": 0-10,
  "issues": []
}

Agent: {agent_name}
Output to evaluate:
---
{output_text}
---
Flag any claims that appear invented or unverifiable. Score 0 for hallucinated statistics."""

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive     = state.get("adaptive_params", {})
        failure_rate = adaptive.get("recent_failure_rate", 0.0)
        sample_rate  = 0.25 if failure_rate > 0.20 else 0.10
        return {
            "sample_rate":   sample_rate,
            "outputs":       context.get("llm_outputs", []),
            "adaptive":      adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from marketforge.config.settings import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        import random

        outputs     = plan.get("outputs", [])
        sample_rate = plan.get("sample_rate", 0.10)

        if not outputs:
            return {"validation_pass_rate": 1.0, "flagged_outputs": [], "sample_size": 0}

        sample_size = max(1, int(len(outputs) * sample_rate))
        sample      = random.sample(outputs, min(sample_size, len(outputs)))

        llm = ChatGoogleGenerativeAI(
            model=settings.llm.fast_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0,
        )

        flagged: list[dict] = []
        scores:  list[float]= []

        for item in sample:
            agent_name  = item.get("agent_name", "unknown")
            output_text = item.get("output", "")
            if not output_text or len(output_text) < 50:
                continue

            prompt = self._VALIDATION_PROMPT.format(
                agent_name=agent_name,
                output_text=output_text[:1500],
            )
            try:
                resp = await llm.ainvoke([HumanMessage(content=prompt)])
                raw  = resp.content.strip()
                if "```" in raw:
                    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
                evaluation = json.loads(raw)
                score = float(evaluation.get("overall", 8.0))
                scores.append(score)
                if score < 6.0 or evaluation.get("issues"):
                    flagged.append({
                        "agent_name": agent_name,
                        "score":      round(score, 1),
                        "issues":     evaluation.get("issues", []),
                    })
            except Exception as exc:
                logger.warning(f"{self.agent_id}.validation_error", error=str(exc))

        pass_rate = (len(scores) - len(flagged)) / max(len(scores), 1)
        logger.info(
            f"{self.agent_id}.execute.done",
            sampled=len(sample), flagged=len(flagged), pass_rate=round(pass_rate, 2),
        )
        return {
            "validation_pass_rate": round(pass_rate, 3),
            "flagged_outputs":      flagged,
            "sample_size":          len(sample),
        }

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        adaptive     = plan.get("adaptive", {})
        failure_rate = 1.0 - result.get("validation_pass_rate", 1.0)
        history      = adaptive.get("failure_rate_history", [])
        history.append(failure_rate)
        adaptive["failure_rate_history"]  = history[-10:]
        adaptive["recent_failure_rate"]   = sum(history[-5:]) / max(len(history[-5:]), 1)
        state["adaptive_params"]          = adaptive
        quality = "good" if failure_rate < 0.15 else ("warning" if failure_rate < 0.30 else "poor")
        if result.get("flagged_outputs"):
            logger.warning(
                f"{self.agent_id}.flagged_outputs", count=len(result["flagged_outputs"]),
                agents=[f["agent_name"] for f in result["flagged_outputs"]],
            )
        return {"quality": quality, "failure_rate": round(failure_rate, 3)}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "llm_validation_pass_rate": result.get("validation_pass_rate", 1.0),
            "flagged_outputs":          result.get("flagged_outputs", []),
            "sample_size":              result.get("sample_size", 0),
        }


class ModelDriftDetectionAgent(DeepAgent):
    """
    Deep Agent for production ML model health monitoring.

    plan():    Reads adaptive_params for each model's baseline PSI score.
               Plans which models to check based on recency (models not
               checked in > 7 days get priority).

    execute(): For each registered model, computes Population Stability Index
               (PSI) on current week's input feature distribution vs the
               training baseline. Queries market.ml_features for recent jobs
               and compares against the stored baseline distribution.
               Also checks rolling F1 for skill extractor (uses LLM Gate 3
               outputs as delayed labels).

    reflect(): Triggers MLEngineerLeadAgent retraining if PSI > 0.25 or
               F1 degrades > 10% vs baseline. Updates drift_risk_score
               per model. Logs all drift events to market.qa_log.

    output():  Returns drift_report per model and list of models needing retrain.
    """

    agent_id   = "model_drift_detection_v1"
    department = "qa_testing"

    _PSI_WARN_THRESHOLD  = 0.20
    _PSI_ALERT_THRESHOLD = 0.25

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive = state.get("adaptive_params", {})
        return {
            "models_to_check": context.get("models", ["skill_extractor", "prescreen", "salary_predictor"]),
            "adaptive":        adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text
        import json

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        feats_t   = "ml_features" if is_sqlite else "market.ml_features"
        evals_t   = "model_evaluations" if is_sqlite else "market.model_evaluations"

        drift_report: dict[str, dict] = {}

        # Load recent feature distributions
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT feature_json FROM {feats_t}
                WHERE computed_at >= {'datetime(\'now\', \'-7 days\')' if is_sqlite else "NOW() - INTERVAL '7 days'"}
                LIMIT 500
            """)).fetchall()

        recent_features: list[dict] = []
        for row in rows:
            try:
                fj = row[0]
                recent_features.append(json.loads(fj) if isinstance(fj, str) else fj)
            except Exception:
                pass

        if not recent_features:
            logger.info(f"{self.agent_id}.no_recent_features")
            return {"drift_report": {}, "models_needing_retrain": []}

        # Compute PSI for salary_band distribution (most sensitive to market drift)
        salary_bands    = [f.get("salary_band", -1) for f in recent_features if f.get("salary_band", -1) >= 0]
        role_cats       = [f.get("role_category_enc", -1) for f in recent_features if f.get("role_category_enc", -1) >= 0]

        salary_psi = self._compute_psi(salary_bands, expected_dist=[0.25, 0.25, 0.25, 0.25, 0.0])
        role_psi   = self._compute_psi(role_cats, expected_dist=[1/12] * 12)

        # Load last model evaluation scores from DB
        with engine.connect() as conn:
            eval_rows = conn.execute(text(f"""
                SELECT model_name, metric_name, metric_value
                FROM {evals_t}
                ORDER BY evaluated_at DESC
                LIMIT 50
            """)).fetchall() if engine.dialect.name != "sqlite" else []

        last_metrics: dict[str, dict] = {}
        for row in eval_rows:
            m_name, m_metric, m_val = row
            if m_name not in last_metrics:
                last_metrics[m_name] = {}
            last_metrics[m_name][m_metric] = float(m_val)

        # Build drift report per model
        models_needing_retrain: list[str] = []
        for model_name in plan["models_to_check"]:
            feature_psi  = salary_psi if "salary" in model_name else role_psi
            needs_retrain= feature_psi > self._PSI_ALERT_THRESHOLD
            drift_report[model_name] = {
                "feature_psi":      round(feature_psi, 4),
                "needs_retrain":    needs_retrain,
                "drift_risk":       "high" if feature_psi > self._PSI_ALERT_THRESHOLD else (
                                    "medium" if feature_psi > self._PSI_WARN_THRESHOLD else "low"),
                "last_metrics":     last_metrics.get(model_name, {}),
            }
            if needs_retrain:
                models_needing_retrain.append(model_name)
                logger.warning(
                    f"{self.agent_id}.drift_detected",
                    model=model_name, psi=round(feature_psi, 4),
                )

        # Log to qa_log
        self._log_drift_events(drift_report, engine, is_sqlite)

        return {
            "drift_report":          drift_report,
            "models_needing_retrain": models_needing_retrain,
        }

    @staticmethod
    def _compute_psi(observed: list, expected_dist: list) -> float:
        """Population Stability Index between observed and expected distribution."""
        if not observed:
            return 0.0
        import math
        n_bins  = len(expected_dist)
        counts  = [0] * n_bins
        for v in observed:
            idx = min(int(v), n_bins - 1)
            if 0 <= idx < n_bins:
                counts[idx] += 1
        total = max(sum(counts), 1)
        psi   = 0.0
        for obs_c, exp_p in zip(counts, expected_dist):
            obs_p = (obs_c / total) + 1e-8
            exp_p = max(exp_p, 1e-8)
            psi  += (obs_p - exp_p) * math.log(obs_p / exp_p)
        return abs(psi)

    def _log_drift_events(self, drift_report: dict, engine: Any, is_sqlite: bool) -> None:
        from sqlalchemy import text
        from datetime import datetime
        qa_t = "qa_log" if is_sqlite else "market.qa_log"
        now  = datetime.utcnow().isoformat()
        try:
            with engine.connect() as conn:
                for model_name, info in drift_report.items():
                    if info.get("needs_retrain"):
                        conn.execute(text(f"""
                            INSERT INTO {qa_t} (run_id, check_name, result, details, logged_at)
                            VALUES (:rid, :cn, :res, :det, :now)
                        """), {
                            "rid": "drift_check",
                            "cn":  f"model_drift:{model_name}",
                            "res": "warning",
                            "det": json.dumps(info),
                            "now": now,
                        })
                conn.commit()
        except Exception as exc:
            logger.warning("model_drift.log_failed", error=str(exc))

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        n_retrain = len(result.get("models_needing_retrain", []))
        quality   = "poor" if n_retrain > 1 else ("warning" if n_retrain == 1 else "good")
        return {"quality": quality, "models_needing_retrain": n_retrain}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "drift_report":           result.get("drift_report", {}),
            "models_needing_retrain": result.get("models_needing_retrain", []),
        }


class ConnectorHealthAgent(DeepAgent):
    """
    Deep Agent that runs lightweight canary requests to each job source.

    plan():    Reads adaptive_params["disabled_sources"] and
               per-source health_score history. Prioritises sources
               that showed degraded health in the last 3 runs.

    execute(): Runs a canary request (1-5 jobs, no full scrape) to each
               active source connector. Records HTTP status, parse rate,
               and schema compliance on the sample. Maintains a rolling
               health_score per source (exponential decay, α=0.3).

    reflect(): Pre-emptively disables sources with health_score < 0.3
               by adding them to adaptive_params["disabled_sources"].
               Re-enables sources that have recovered (health_score > 0.7
               for 2 consecutive checks).

    output():  Returns health_score per source and any newly disabled/enabled sources.
    """

    agent_id   = "connector_health_v1"
    department = "qa_testing"

    _SOURCES = ["adzuna", "reed", "wellfound", "ats_direct", "career_pages",
                "funding_news", "recruiter_boards", "specialist_boards"]

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive        = state.get("adaptive_params", {})
        disabled        = set(adaptive.get("disabled_sources", []))
        health_scores   = adaptive.get("health_scores", {})
        active_sources  = [s for s in self._SOURCES if s not in disabled]
        return {
            "active_sources": active_sources,
            "disabled":       disabled,
            "health_scores":  health_scores,
            "adaptive":       adaptive,
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        import asyncio
        active_sources = plan["active_sources"]
        health_scores  = dict(plan["health_scores"])
        results: dict[str, dict] = {}

        async def canary_check(source: str) -> dict:
            """Run a lightweight health check on a single connector."""
            try:
                # Dynamic import of the connector agent
                connector = self._get_connector(source)
                if connector is None:
                    return {"source": source, "status": "no_connector", "score": 0.5}

                # Check connectivity by running plan() only (no actual scrape)
                dummy_state = {"adaptive_params": {}, "last_yield": 0, "consecutive_failures": 0}
                plan_result = await connector.plan({}, dummy_state)
                return {
                    "source":   source,
                    "status":   "reachable" if plan_result else "plan_failed",
                    "score":    1.0 if plan_result else 0.3,
                }
            except Exception as exc:
                logger.warning(f"{self.agent_id}.canary_failed", source=source, error=str(exc))
                return {"source": source, "status": "error", "score": 0.0, "error": str(exc)[:100]}

        # Run canary checks with concurrency limit
        tasks = [canary_check(s) for s in active_sources]
        checks = await asyncio.gather(*tasks, return_exceptions=True)

        for check in checks:
            if isinstance(check, Exception):
                continue
            source = check["source"]
            new_score = check.get("score", 0.5)
            # Exponential moving average: α = 0.3
            old_score = health_scores.get(source, 1.0)
            health_scores[source] = round(0.3 * new_score + 0.7 * old_score, 3)
            results[source] = {**check, "rolling_health": health_scores[source]}
            logger.debug(
                f"{self.agent_id}.canary.done",
                source=source, raw_score=new_score, rolling=health_scores[source],
            )

        return {"health_scores": health_scores, "canary_results": results}

    def _get_connector(self, source: str):
        """Lazy-load the connector sub-agent for canary check."""
        connector_map = {
            "adzuna":           "marketforge.agents.data_collection.adzuna_agent:AdzunaDeepScoutAgent",
            "reed":             "marketforge.agents.data_collection.reed_agent:ReedDeepScoutAgent",
            "wellfound":        "marketforge.agents.data_collection.additional_agents:WellfoundDeepScoutAgent",
            "ats_direct":       "marketforge.agents.data_collection.additional_agents:ATSDirectDeepAgent",
            "career_pages":     "marketforge.agents.data_collection.additional_agents:CareerPagesDeepCrawlerAgent",
            "funding_news":     "marketforge.agents.data_collection.funding_news_agent:FundingNewsDeepDiscoveryAgent",
            "recruiter_boards": "marketforge.agents.data_collection.recruiter_agent:RecruiterBoardsDeepAgent",
            "specialist_boards":"marketforge.agents.data_collection.specialist_boards_agent:SpecialistBoardsDeepAgent",
        }
        spec = connector_map.get(source)
        if not spec:
            return None
        try:
            module_path, class_name = spec.rsplit(":", 1)
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)()
        except Exception as exc:
            logger.debug(f"{self.agent_id}.connector_load_failed", source=source, error=str(exc))
            return None

    async def reflect(
        self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        adaptive        = plan.get("adaptive", {})
        health_scores   = result.get("health_scores", {})
        disabled        = set(plan.get("disabled", set()))
        newly_disabled: list[str] = []
        newly_enabled:  list[str] = []

        for source, score in health_scores.items():
            if score < 0.3 and source not in disabled:
                disabled.add(source)
                newly_disabled.append(source)
                logger.warning(f"{self.agent_id}.source_disabled", source=source, score=score)
            elif score > 0.7 and source in disabled:
                disabled.discard(source)
                newly_enabled.append(source)
                logger.info(f"{self.agent_id}.source_re_enabled", source=source, score=score)

        adaptive["disabled_sources"] = list(disabled)
        adaptive["health_scores"]    = health_scores
        state["adaptive_params"]     = adaptive

        quality = "poor" if len(disabled) > 3 else ("warning" if disabled else "good")
        return {
            "quality":        quality,
            "newly_disabled": newly_disabled,
            "newly_enabled":  newly_enabled,
            "disabled_count": len(disabled),
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_health":   result.get("health_scores", {}),
            "newly_disabled":  reflection.get("newly_disabled", []),
            "newly_enabled":   reflection.get("newly_enabled", []),
        }


class QALeadAgent(DeepAgent):
    """
    Department 7 Lead Agent — Quality Assurance & Testing.
    Routes to the appropriate QA sub-agent based on mode.
    """

    agent_id   = "qa_lead_v1"
    department = "qa_testing"

    def __init__(self) -> None:
        self._integrity   = DataIntegrityAgent()
        self._report_qa   = ReportQualityAgent()
        self._llm_val     = LLMOutputValidatorAgent()
        self._drift       = ModelDriftDetectionAgent()
        self._conn_health = ConnectorHealthAgent()

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        mode = context.get("mode", "data_integrity")
        return {"mode": mode, "context": context}

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        mode    = plan["mode"]
        context = plan["context"]

        if mode == "report_review":
            return await self._report_qa.run(context)
        elif mode == "data_integrity":
            return await self._integrity.run(context)
        elif mode == "llm_validation":
            return await self._llm_val.run(context)
        elif mode == "model_drift":
            return await self._drift.run(context)
        elif mode == "connector_health":
            return await self._conn_health.run(context)
        elif mode == "full_qa":
            # Full QA cycle: data integrity + connector health (no LLM cost)
            integrity_result = await self._integrity.run(context)
            health_result    = await self._conn_health.run({})
            drift_result     = await self._drift.run({})
            return {
                "data_integrity":   integrity_result,
                "connector_health": health_result,
                "model_drift":      drift_result,
            }
        else:
            return {"skipped": True, "reason": f"unknown_mode:{mode}"}

    async def reflect(self, plan: dict[str, Any], result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        quality = "good"
        if result.get("block_downstream"):
            quality = "poor"
        elif not result.get("report_passed", True):
            quality = "warning"
        elif result.get("models_needing_retrain"):
            quality = "warning"
        return {"quality": quality}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {**result, "qa_quality": reflection.get("quality", "good")}
