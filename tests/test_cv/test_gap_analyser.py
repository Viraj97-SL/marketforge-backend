"""
MarketForge AI — CV Gap Analyser Tests.

Regression tests for the gap-list contradiction bug: a Gate-3 (LLM) paraphrase
of a skill already on the CV ("Retrieval-Augmented Generation" vs. the CV's
"RAG") must not surface as a gap, near-duplicate concepts under two different
surface forms must not both appear, and broad umbrella labels ("Machine
Learning") must not be listed once the CV already demonstrates the concept
through specific tools.

analyse_gaps() hits the DB via _fetch_market_data() — monkeypatched here so
these tests exercise the dedup/ranking logic in isolation, no DB required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

os.environ.setdefault("DATABASE_URL_SYNC", "sqlite:///./test_gap_analyser.db")

import pytest

import marketforge.cv.gap_analyser as gap_analyser
from marketforge.cv.gap_analyser import analyse_gaps


def _mock_market(monkeypatch, top_skills: dict[str, int], rising: list[str] | None = None) -> None:
    monkeypatch.setattr(
        gap_analyser, "_fetch_market_data",
        lambda target_role="": {"top_skills": top_skills, "rising_skills": rising or []},
    )


class TestCanonicalDedupAgainstCV:
    def test_gate3_paraphrase_of_cv_skill_is_not_a_gap(self, monkeypatch):
        # Bug: found "RAG" -> gap "Retrieval-Augmented Generation"
        _mock_market(monkeypatch, {"Retrieval-Augmented Generation": 100, "Docker": 80})
        result = analyse_gaps(cv_skills=["RAG"], target_role="ml_engineer")
        skills = [g.skill for g in result.all_gaps]
        assert "Retrieval-Augmented Generation" not in skills
        assert "Docker" in skills

    def test_gate3_alias_of_cv_skill_is_not_a_gap(self, monkeypatch):
        # Bug: found "LangGraph", "Multi-agent systems" -> gap "Agentic AI / Machine Learning"
        # "Agentic AI" alone is an existing alias of "Multi-agent systems".
        _mock_market(monkeypatch, {"Agentic AI": 100, "Docker": 80})
        result = analyse_gaps(cv_skills=["Multi-agent systems"], target_role="ml_engineer")
        skills = [g.skill for g in result.all_gaps]
        assert "Agentic AI" not in skills
        assert "Docker" in skills


class TestSelfDedupWithinGapList:
    def test_two_surface_forms_of_one_concept_collapse_to_one_gap(self, monkeypatch):
        # Bug: gap list contains both "GenAI" and "Generative AI"
        _mock_market(monkeypatch, {"GenAI": 100, "Generative AI": 90, "Docker": 80})
        result = analyse_gaps(cv_skills=[], target_role="ml_engineer")
        skills = [g.skill for g in result.all_gaps]
        genai_variants = [s for s in skills if s in ("GenAI", "Generative AI")]
        assert len(genai_variants) == 1


class TestUmbrellaSuppression:
    def test_machine_learning_suppressed_when_cv_shows_specific_ml_tools(self, monkeypatch):
        # Bug: found "PyTorch", "scikit-learn", "XGBoost", "LightGBM" -> gap "Machine Learning"
        _mock_market(monkeypatch, {"Machine Learning": 100, "Docker": 80})
        result = analyse_gaps(
            cv_skills=["PyTorch", "scikit-learn", "XGBoost", "LightGBM"],
            target_role="ml_engineer",
        )
        skills = [g.skill for g in result.all_gaps]
        assert "Machine Learning" not in skills
        assert "Docker" in skills

    def test_machine_learning_still_shown_when_cv_has_no_specific_ml_tools(self, monkeypatch):
        # Umbrella suppression must not swallow a genuine gap for a CV with
        # no ML tooling at all.
        _mock_market(monkeypatch, {"Machine Learning": 100})
        result = analyse_gaps(cv_skills=["Docker"], target_role="ml_engineer")
        skills = [g.skill for g in result.all_gaps]
        assert "Machine Learning" in skills

    def test_artificial_intelligence_suppressed_when_cv_shows_llm_and_agent_work(self, monkeypatch):
        _mock_market(monkeypatch, {"Artificial Intelligence": 100, "Docker": 80})
        result = analyse_gaps(
            cv_skills=["LangGraph", "Multi-agent systems"],
            target_role="ml_engineer",
        )
        skills = [g.skill for g in result.all_gaps]
        assert "Artificial Intelligence" not in skills
        assert "Docker" in skills


class TestCompoundGate3StringSplitting:
    def test_compound_string_split_and_each_part_checked_against_cv(self, monkeypatch):
        # Real observed bug: Gate 3 returned "Agentic AI / Machine Learning"
        # as ONE string, which never matched any single alias and slipped
        # past canonicalisation whole. Both halves are contradictions here:
        # "Agentic AI" is an alias of "Multi-agent systems" (found), and
        # "Machine Learning" is implied by the found PyTorch + scikit-learn.
        _mock_market(monkeypatch, {"Agentic AI / Machine Learning": 100, "Docker": 80})
        result = analyse_gaps(
            cv_skills=["Multi-agent systems", "PyTorch", "scikit-learn"],
            target_role="ml_engineer",
        )
        skills = [g.skill for g in result.all_gaps]
        assert not any("Agentic AI" in s or "Machine Learning" in s for s in skills)
        assert "Docker" in skills

    def test_llm_umbrella_suppressed_when_cv_shows_llm_tools(self, monkeypatch):
        _mock_market(monkeypatch, {"LLM": 100, "Docker": 80})
        result = analyse_gaps(
            cv_skills=["OpenAI API", "Hugging Face"],
            target_role="ml_engineer",
        )
        skills = [g.skill for g in result.all_gaps]
        assert "LLM" not in skills
        assert "Docker" in skills


class TestBaselineBehaviourUnaffected:
    def test_genuinely_missing_specific_skill_still_appears(self, monkeypatch):
        _mock_market(monkeypatch, {"Kubernetes": 100})
        result = analyse_gaps(cv_skills=["Python"], target_role="ml_engineer")
        skills = [g.skill for g in result.all_gaps]
        assert "Kubernetes" in skills

    def test_no_market_data_returns_empty_analysis(self, monkeypatch):
        monkeypatch.setattr(gap_analyser, "_fetch_market_data", lambda target_role="": None)
        result = analyse_gaps(cv_skills=["Python"], target_role="ml_engineer")
        assert result.all_gaps == []


@pytest.fixture()
def salary_db(tmp_path):
    """Real sqlite DB seeded with per-job salary data, for testing the
    per-skill salary uplift query end-to-end (not monkeypatched)."""
    db_path    = str(tmp_path / "salary_gap.db")
    sqlite_url = f"sqlite:///{db_path}"

    from marketforge.memory import postgres
    from marketforge.config.settings import settings as _settings
    from sqlalchemy import text

    old_engine   = postgres._sync_engine
    old_sync_url = _settings.database_url_sync

    postgres._sync_engine       = None
    _settings.database_url_sync = sqlite_url
    os.environ["DATABASE_URL_SYNC"] = sqlite_url

    from marketforge.memory.postgres import init_database, get_sync_engine
    init_database()

    engine = get_sync_engine()
    with engine.connect() as conn:
        # 12 "Docker" postings at a high salary, 12 baseline postings at a
        # lower salary — Docker should measure a real >1.0 uplift.
        for i in range(12):
            job_id = f"docker-{i}"
            conn.execute(text(
                "INSERT INTO jobs (job_id, dedup_hash, run_id, title, company, role_category, source, salary_midpoint) "
                "VALUES (:jid, :jid, 'run1', 'ML Engineer', 'Acme', 'ml_engineer', 'test', 90000)"
            ), {"jid": job_id})
            conn.execute(text(
                "INSERT INTO job_skills (job_id, skill, extraction_method) VALUES (:jid, 'Docker', 'gate1')"
            ), {"jid": job_id})
        for i in range(12):
            job_id = f"base-{i}"
            conn.execute(text(
                "INSERT INTO jobs (job_id, dedup_hash, run_id, title, company, role_category, source, salary_midpoint) "
                "VALUES (:jid, :jid, 'run1', 'ML Engineer', 'Acme', 'ml_engineer', 'test', 60000)"
            ), {"jid": job_id})
            conn.execute(text(
                "INSERT INTO job_skills (job_id, skill, extraction_method) VALUES (:jid, 'Rust', 'gate1')"
            ), {"jid": job_id})
        # Only 2 postings mention "Kubeflow" — below MIN_SALARY_SAMPLE_SIZE,
        # must fall back to the rank estimate rather than report a "median".
        for i in range(2):
            job_id = f"kubeflow-{i}"
            conn.execute(text(
                "INSERT INTO jobs (job_id, dedup_hash, run_id, title, company, role_category, source, salary_midpoint) "
                "VALUES (:jid, :jid, 'run1', 'ML Engineer', 'Acme', 'ml_engineer', 'test', 150000)"
            ), {"jid": job_id})
            conn.execute(text(
                "INSERT INTO job_skills (job_id, skill, extraction_method) VALUES (:jid, 'Kubeflow', 'gate1')"
            ), {"jid": job_id})
        conn.commit()

    yield

    if postgres._sync_engine is not None:
        postgres._sync_engine.dispose()
    postgres._sync_engine       = None
    _settings.database_url_sync = old_sync_url
    os.environ["DATABASE_URL_SYNC"] = old_sync_url
    if old_engine is not None:
        postgres._sync_engine = old_engine


class TestSalaryUplift:
    def test_high_salary_skill_measured_with_sufficient_sample(self, salary_db):
        result = analyse_gaps(cv_skills=[], target_role="ml_engineer")
        docker_gap = next(g for g in result.all_gaps if g.skill == "Docker")
        assert docker_gap.salary_basis == "measured"

    def test_thin_sample_skill_falls_back_to_rank_estimate(self, salary_db):
        result = analyse_gaps(cv_skills=[], target_role="ml_engineer")
        kubeflow_gap = next(g for g in result.all_gaps if g.skill == "Kubeflow")
        assert kubeflow_gap.salary_basis == "rank_estimate"
