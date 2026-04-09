"""
MarketForge AI — CV Gap Analyser.

ML-ranked skill gap prioritisation using market demand data + salary uplift.
No LLM required — fully deterministic and auditable.

Priority score formula:
  priority = demand_score × salary_uplift_factor × recency_weight

Where:
  demand_score        = normalised rank in top_skills (0–1, higher = more in demand)
  salary_uplift_factor= ratio of median salary for jobs requiring this skill
                        vs overall median (estimated from co-occurrence patterns)
  recency_weight      = bonus for skills in rising_skills list (1.2×)

Output groups gaps into:
  short_term  — top skills addressable in 0–3 months (certifications, courses)
  mid_term    — 3–12 months (projects, bootcamps, work experience)
  long_term   — 12+ months (advanced degrees, deep specialisations)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# ── Role normalisation (mirrors ats_scorer) ───────────────────────────────────
_ROLE_MAP: dict[str, str] = {
    "ml engineer":                "ml_engineer",
    "data scientist":             "data_scientist",
    "ai engineer":                "ai_engineer",
    "mlops engineer":             "mlops_engineer",
    "nlp engineer":               "nlp_engineer",
    "computer vision engineer":   "computer_vision_engineer",
    "research scientist":         "research_scientist",
    "applied scientist":          "applied_scientist",
    "data engineer":              "data_engineer",
    "ai safety researcher":       "ai_safety_researcher",
    "ai product manager":         "ai_product_manager",
    "ml_engineer":                "ml_engineer",
    "data_scientist":             "data_scientist",
    "ai_engineer":                "ai_engineer",
    "mlops_engineer":             "mlops_engineer",
    "nlp_engineer":               "nlp_engineer",
    "computer_vision_engineer":   "computer_vision_engineer",
    "research_scientist":         "research_scientist",
    "applied_scientist":          "applied_scientist",
    "data_engineer":              "data_engineer",
    "ai_safety_researcher":       "ai_safety_researcher",
    "ai_product_manager":         "ai_product_manager",
}


def _normalise_role(target_role: str) -> str:
    return _ROLE_MAP.get(target_role.lower().strip(), "other")


# ── Skills addressable quickly (courses / certs exist) ────────────────────────
_SHORT_TERM_SKILLS: frozenset[str] = frozenset({
    "Docker", "Kubernetes", "MLflow", "Airflow", "FastAPI", "Flask",
    "SQL", "PostgreSQL", "Redis", "Git", "CI/CD", "Terraform",
    "scikit-learn", "Pandas", "NumPy", "Matplotlib", "Plotly",
    "AWS", "GCP", "Azure", "REST API", "LangChain", "LangGraph",
    "Hugging Face", "spaCy", "NLTK", "OpenCV", "Weights & Biases",
    "Ray", "Dask", "PySpark", "Apache Spark", "dbt",
})

# ── Skills requiring sustained deep work (12+ months) ────────────────────────
_LONG_TERM_SKILLS: frozenset[str] = frozenset({
    "PyTorch", "TensorFlow", "JAX", "CUDA", "C++", "Rust",
    "Reinforcement Learning", "RLHF", "Diffusion Models",
    "Transformer Architecture", "Custom CUDA Kernels",
    "Research Publications", "PhD-level Research",
    "Distributed Training", "Systems Programming",
})


@dataclass
class SkillGap:
    skill:               str
    category:            str | None
    demand_score:        float      # 0–1, normalised market demand rank
    priority_score:      float      # final weighted score (higher = act sooner)
    is_rising:           bool       # true if in rising_skills list
    time_horizon:        str        # "short" | "mid" | "long"
    market_demand_count: int        # raw count from weekly snapshot


@dataclass
class GapAnalysis:
    short_term: list[SkillGap] = field(default_factory=list)   # 0–3 months
    mid_term:   list[SkillGap] = field(default_factory=list)   # 3–12 months
    long_term:  list[SkillGap] = field(default_factory=list)   # 12+ months

    @property
    def all_gaps(self) -> list[SkillGap]:
        return self.short_term + self.mid_term + self.long_term

    def top_n(self, n: int = 10) -> list[SkillGap]:
        """Return top-N gaps by priority score across all time horizons."""
        return sorted(self.all_gaps, key=lambda g: -g.priority_score)[:n]


def analyse_gaps(
    cv_skills:   list[str],
    target_role: str,
    top_n:       int = 15,
) -> GapAnalysis:
    """
    Compute prioritised skill gaps between CV and market demand.

    Args:
        cv_skills:   Skills extracted from the uploaded CV.
        target_role: User's target role (used for market lookup).
        top_n:       Maximum number of gaps to return.

    Returns:
        GapAnalysis with gaps bucketed into short/mid/long-term horizons.
    """
    market_data = _fetch_market_data(target_role)
    if not market_data:
        return GapAnalysis()

    top_skills    = market_data["top_skills"]       # {skill: count}
    rising_skills = set(market_data.get("rising_skills", []))
    cv_lower      = {s.lower() for s in cv_skills}

    if not top_skills:
        return GapAnalysis()

    # ── Normalise demand scores ────────────────────────────────────────────────
    max_count = max(top_skills.values()) or 1
    gaps: list[SkillGap] = []

    for rank, (skill, count) in enumerate(
        sorted(top_skills.items(), key=lambda x: -x[1])[:top_n * 2]
    ):
        if skill.lower() in cv_lower:
            continue   # user already has this skill

        demand_score   = count / max_count
        recency_weight = 1.2 if skill in rising_skills else 1.0

        # Salary uplift estimate: skills in top 10 demand are assumed to carry
        # a 10–20% salary premium (crude but deterministic without per-skill data)
        salary_factor  = 1.0 + max(0.0, (1.0 - rank / len(top_skills)) * 0.2)

        priority_score = demand_score * salary_factor * recency_weight

        time_horizon = _classify_horizon(skill)

        gaps.append(SkillGap(
            skill               = skill,
            category            = _infer_category(skill),
            demand_score        = round(demand_score, 3),
            priority_score      = round(priority_score, 3),
            is_rising           = skill in rising_skills,
            time_horizon        = time_horizon,
            market_demand_count = count,
        ))

    # Sort by priority descending, then slice
    gaps.sort(key=lambda g: -g.priority_score)
    gaps = gaps[:top_n]

    # Bucket into time horizons
    result = GapAnalysis()
    for gap in gaps:
        if gap.time_horizon == "short":
            result.short_term.append(gap)
        elif gap.time_horizon == "long":
            result.long_term.append(gap)
        else:
            result.mid_term.append(gap)

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_market_data(target_role: str = "") -> dict | None:
    """
    Pull role-specific skill demand from DB.

    Query strategy (most specific → least specific):
      1. Live job_skills JOIN jobs WHERE role_category = <normalised_role>
      2. weekly_snapshots WHERE role_category = <normalised_role>
      3. weekly_snapshots WHERE role_category = 'all'
    """
    try:
        import json
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        role_cat  = _normalise_role(target_role) if target_role else "other"
        jobs_t    = "jobs"       if is_sqlite else "market.jobs"
        skills_t  = "job_skills" if is_sqlite else "market.job_skills"
        snap_t    = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

        top_skills: dict[str, int] = {}
        rising_skills: list[str]   = []

        with engine.connect() as conn:
            # ── Strategy 1: live per-role aggregation ─────────────────────────
            rows = conn.execute(text(f"""
                SELECT js.skill, COUNT(*) AS cnt
                FROM {skills_t} js
                JOIN {jobs_t} j ON j.job_id = js.job_id
                WHERE j.role_category = :role
                GROUP BY js.skill
                ORDER BY cnt DESC
                LIMIT 60
            """), {"role": role_cat}).fetchall()

            if rows:
                top_skills = {r[0]: r[1] for r in rows}
                logger.debug("gap_analyser.role_live", role=role_cat, skills=len(top_skills))
            else:
                # ── Strategy 2: role-specific snapshot ────────────────────────
                row = conn.execute(text(f"""
                    SELECT top_skills, rising_skills
                    FROM {snap_t}
                    WHERE role_category = :role
                    ORDER BY week_start DESC LIMIT 1
                """), {"role": role_cat}).fetchone()

                if row and row[0]:
                    top_skills    = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
                    rising_skills = json.loads(row[1]) if isinstance(row[1], str) else (row[1] or [])
                    logger.debug("gap_analyser.role_snapshot", role=role_cat)

            if not top_skills:
                # ── Strategy 3: global 'all' snapshot fallback ────────────────
                row = conn.execute(text(f"""
                    SELECT top_skills, rising_skills
                    FROM {snap_t}
                    WHERE role_category = 'all'
                    ORDER BY week_start DESC LIMIT 1
                """)).fetchone()
                if row and row[0]:
                    top_skills    = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
                    rising_skills = json.loads(row[1]) if isinstance(row[1], str) else (row[1] or [])
                    logger.debug("gap_analyser.global_fallback")

        if not top_skills:
            return None

        return {"top_skills": top_skills, "rising_skills": rising_skills}

    except Exception as exc:
        logger.warning("gap_analyser.fetch_error", error=str(exc))
        return None


def _classify_horizon(skill: str) -> str:
    """Assign a time horizon based on known skill learning curves."""
    if skill in _SHORT_TERM_SKILLS:
        return "short"
    if skill in _LONG_TERM_SKILLS:
        return "long"
    return "mid"


def _infer_category(skill: str) -> str | None:
    """Lightweight category inference — mirrors the taxonomy where possible."""
    _CATS = {
        "language":        {"Python", "R", "SQL", "C++", "Java", "Go", "Rust", "Scala", "Julia"},
        "dl_framework":    {"PyTorch", "TensorFlow", "JAX", "Keras", "MXNet"},
        "ml_library":      {"scikit-learn", "XGBoost", "LightGBM", "CatBoost", "statsmodels"},
        "llm_framework":   {"LangChain", "LangGraph", "LlamaIndex", "Hugging Face", "OpenAI SDK"},
        "infra":           {"Docker", "Kubernetes", "Terraform", "AWS", "GCP", "Azure"},
        "mlops":           {"MLflow", "Weights & Biases", "DVC", "Airflow", "Kubeflow", "Ray"},
        "data_engineering":{"Apache Spark", "dbt", "Kafka", "Flink", "Databricks"},
        "backend":         {"FastAPI", "Flask", "Django", "REST API", "GraphQL"},
    }
    for cat, skills in _CATS.items():
        if skill in skills:
            return cat
    return None
