"""
MarketForge AI — Core Domain Models.

All market-facing data flows through these typed models.
Personal scoring or CV data is explicitly absent — this is a market intelligence
platform, not a job-matching tool.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field


# ── Role category taxonomy ────────────────────────────────────────────────────
RoleCategory = Literal[
    "ml_engineer",
    "data_scientist",
    "ai_engineer",
    "mlops_engineer",
    "nlp_engineer",
    "computer_vision_engineer",
    "research_scientist",
    "applied_scientist",
    "data_engineer",
    "ai_safety_researcher",
    "ai_product_manager",
    "other",
]

ExperienceLevel = Literal["junior", "mid", "senior", "lead", "principal", "unknown"]
WorkModel       = Literal["remote", "hybrid", "onsite", "unknown"]
CompanyStage    = Literal[
    "seed", "series_a", "series_b", "series_c", "series_d_plus",
    "growth", "enterprise", "nonprofit", "government",
    "recently_funded", "acquired", "public", "unknown",
]


class RawJob(BaseModel):
    """
    A job posting as returned by any data collection sub-agent.
    Source-agnostic, fully normalised schema.
    """
    job_id:   str = Field(description="Source-prefixed unique ID, e.g. 'adzuna_12345678'")
    title:    str
    company:  str
    location: str

    # ── Compensation ──────────────────────────────────────────────────────────
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str = "GBP"

    # ── Content ───────────────────────────────────────────────────────────────
    description: Annotated[str, Field(max_length=12_000)] = ""
    url:         str
    source:      str   # connector name: adzuna / reed / wellfound / ats_direct / …

    # ── Classification (populated by NLP pipeline after ingestion) ────────────
    role_category:    RoleCategory    | None = None
    experience_level: ExperienceLevel | None = None
    work_model:       WorkModel              = "unknown"
    industry:         str | None = None

    # ── Company signals ────────────────────────────────────────────────────────
    company_stage:         CompanyStage = "unknown"
    is_startup:            bool         = False
    is_uk_headquartered:   bool | None  = None
    employee_count_band:   str | None   = None   # "1-10" / "11-50" / etc.

    # ── Visa & sponsorship (populated by NLP + rule-based detection) ───────────
    offers_sponsorship:   bool | None  = None
    citizens_only:        bool | None  = None
    sponsorship_signals:  list[str]    = Field(default_factory=list)

    # ── Degree requirement ─────────────────────────────────────────────────────
    degree_required: Literal["bsc", "msc", "phd", "any_degree", "not_required", "not_stated"] = "not_stated"

    # ── Benefits signals ───────────────────────────────────────────────────────
    equity_offered:   bool | None = None
    flexible_hours:   bool | None = None

    # ── Provenance ────────────────────────────────────────────────────────────
    posted_date: date    | None = None
    scraped_at:  datetime        = Field(default_factory=datetime.utcnow)

    # ── Computed ─────────────────────────────────────────────────────────────
    @computed_field
    @property
    def dedup_hash(self) -> str:
        """
        Stable 16-char hash for cross-source, cross-run deduplication.
        Based on normalised (title, company, location) — ignores description
        changes on the same physical role.
        """
        raw = "|".join([
            self.title.lower().strip(),
            self.company.lower().strip(),
            self.location.lower().strip(),
        ])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @computed_field
    @property
    def salary_display(self) -> str:
        if self.salary_min and self.salary_max:
            return f"£{self.salary_min:,.0f}–£{self.salary_max:,.0f}"
        elif self.salary_min:
            return f"From £{self.salary_min:,.0f}"
        elif self.salary_max:
            return f"Up to £{self.salary_max:,.0f}"
        return "Not disclosed"

    @computed_field
    @property
    def salary_midpoint(self) -> float | None:
        if self.salary_min and self.salary_max:
            return (self.salary_min + self.salary_max) / 2
        return self.salary_max or self.salary_min


class EnrichedJob(RawJob):
    """
    A job after the NLP pipeline has run — skills extracted, category confirmed.
    Written back to the database after the nlp_extraction Airflow task completes.
    """
    extracted_skills:  list[str]  = Field(default_factory=list)
    skill_categories:  dict[str, str] = Field(default_factory=dict)  # skill → category
    nlp_version:       str = "0.1.0"
    extraction_method: str = "gate1"   # gate1 / gate2 / gate3 / combined


class MarketSnapshot(BaseModel):
    """
    Aggregated weekly market statistics produced by the Market Analysis department.
    Written to market.weekly_snapshots; read by dashboard and Content Studio.
    """
    snapshot_id:      int | None    = None
    week_start:       date
    role_category:    str           = "all"

    # ── Skill stats ───────────────────────────────────────────────────────────
    top_skills:        dict[str, int]   = Field(default_factory=dict)  # skill → count
    rising_skills:     list[str]        = Field(default_factory=list)
    declining_skills:  list[str]        = Field(default_factory=list)

    # ── Salary stats (£) ──────────────────────────────────────────────────────
    salary_p10: float | None = None
    salary_p25: float | None = None
    salary_p50: float | None = None
    salary_p75: float | None = None
    salary_p90: float | None = None
    salary_sample_size: int = 0

    # ── Volume stats ──────────────────────────────────────────────────────────
    job_count:         int   = 0
    new_job_count:     int   = 0
    sponsorship_rate:  float = 0.0   # 0..1
    remote_rate:       float = 0.0
    hybrid_rate:       float = 0.0
    startup_rate:      float = 0.0

    # ── Geographic ────────────────────────────────────────────────────────────
    top_cities:   dict[str, int] = Field(default_factory=dict)

    computed_at:  datetime = Field(default_factory=datetime.utcnow)


class ResearchSignal(BaseModel):
    """
    An emerging technique / technology detected by the Research department.
    Signals which topics are trending in papers before appearing in JDs.
    """
    signal_id:         int | None = None
    technique_name:    str
    source:            Literal["arxiv", "tech_blog", "industry_report"]
    first_seen:        date
    mention_count:     int = 1
    first_in_jd:       date | None = None       # when it first appeared in job postings
    adoption_lag_days: int | None = None        # first_in_jd - first_seen
    relevance_score:   float = 0.0             # 0..1
    arxiv_ids:         list[str] = Field(default_factory=list)
    summary:           str = ""


class PipelineRun(BaseModel):
    """Telemetry record for a single DAG execution."""
    run_id:        str
    dag_name:      str
    started_at:    datetime
    completed_at:  datetime | None = None
    status:        Literal["running", "success", "partial", "failed"] = "running"

    jobs_scraped:       int        = 0
    jobs_new:           int        = 0
    jobs_deduplicated:  int        = 0   # how many were filtered out as dupes
    jobs_enriched:      int        = 0   # how many had NLP skills extracted
    llm_cost_usd:       float      = 0.0
    sources_used:       list[str]  = Field(default_factory=list)   # connector names that ran
    errors:             list[dict] = Field(default_factory=list)   # non-fatal errors captured
    metadata:           dict       = Field(default_factory=dict)


class AgentRunState(BaseModel):
    """
    Persistent state for a single sub-agent, stored in market.agent_state.
    Read by plan() at the start of every run; written by reflect() at the end.
    """
    agent_id:            str
    department:          str
    last_run_at:         datetime | None = None
    last_yield:          int             = 0
    consecutive_failures: int            = 0
    run_count:           int             = 0

    # JSONB-serialised adaptive parameters the agent modifies itself
    adaptive_params:     dict = Field(default_factory=dict)

    # Last 10 reflect() outputs kept for trend detection
    reflection_log:      list[dict] = Field(default_factory=list)

    def record_reflection(self, reflection: dict) -> None:
        """Append a reflection entry and cap the log at 10 entries."""
        self.reflection_log.append(reflection)
        if len(self.reflection_log) > 10:
            self.reflection_log = self.reflection_log[-10:]
