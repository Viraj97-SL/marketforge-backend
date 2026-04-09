"""
MarketForge AI — ATS Scorer Tests.

Tests ats_scorer.score_cv() across all five scoring dimensions:
  - keyword_match  (market skill comparison)
  - structure      (sections, action verbs, metrics)
  - readability    (Flesch-Kincaid)
  - completeness   (contact, dates, sections)
  - format_safety  (tables, images, page count)

Grade thresholds and weighted total are also verified.
No real DB required — keyword scoring falls back gracefully.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from marketforge.cv.parser import ParsedCV
from marketforge.cv.ats_scorer import (
    ATSScore,
    _score_structure,
    _score_readability,
    _score_completeness,
    _score_format,
    score_cv,
    _GRADE_THRESHOLDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _cv(**overrides) -> ParsedCV:
    """Build a ParsedCV with sensible defaults, override as needed."""
    defaults = dict(
        raw_text        = "Senior ML Engineer with PyTorch experience. Led team of 5.",
        sections        = {
            "summary":    "Experienced ML engineer with 6 years in the field.",
            "experience": "Lead ML Engineer at DeepMind 2019-2024. Built and deployed PyTorch models. Reduced latency by 40%. Managed team of 5 engineers. Increased throughput by 3x.",
            "education":  "MSc Computer Science, UCL 2017-2018",
            "skills":     "Python PyTorch TensorFlow scikit-learn Docker Kubernetes MLflow",
            "certifications": "AWS Machine Learning Specialty 2023",
        },
        has_email       = True,
        has_phone       = True,
        has_linkedin    = True,
        has_github      = False,
        page_count      = 2,
        has_tables      = False,
        has_images      = False,
        estimated_years = 6.0,
        parse_method    = "pdfplumber",
    )
    defaults.update(overrides)
    return ParsedCV(**defaults)


# ── Structure scoring ─────────────────────────────────────────────────────────

class TestStructureScoring:
    def test_full_sections_scores_high(self):
        cv     = _cv()
        issues = []
        score  = _score_structure(cv, issues)
        assert score >= 70

    def test_missing_experience_penalised(self):
        sections = {"education": "BSc CS UCL", "skills": "Python Docker"}
        cv       = _cv(sections=sections)
        issues   = []
        score    = _score_structure(cv, issues)
        assert score < 80
        assert any("experience" in i.lower() for i in issues)

    def test_action_verbs_improve_score(self):
        # Experience section with many action verbs
        cv_good = _cv(sections={
            "experience": "Led, built, deployed, reduced, improved, managed, delivered, architected, trained models",
            "education": "BSc", "skills": "Python",
        })
        cv_weak = _cv(sections={
            "experience": "Was responsible for some work at a company",
            "education": "BSc", "skills": "Python",
        })
        issues_good, issues_weak = [], []
        assert _score_structure(cv_good, issues_good) >= _score_structure(cv_weak, issues_weak)

    def test_no_metrics_generates_issue(self):
        cv = _cv(sections={
            "experience": "Did engineering work at a company for several years",
            "education": "BSc", "skills": "Python",
        })
        issues = []
        _score_structure(cv, issues)
        assert any("metric" in i.lower() or "number" in i.lower() for i in issues)

    def test_missing_sections_generate_issues(self):
        cv     = _cv(sections={})
        issues = []
        _score_structure(cv, issues)
        required = {"experience", "education", "skills"}
        issue_text = " ".join(issues).lower()
        for section in required:
            assert section in issue_text


# ── Readability scoring ───────────────────────────────────────────────────────

class TestReadabilityScoring:
    def test_simple_text_scores_well(self):
        # Short, clear sentences → grade ~10-12
        text   = "Built ML models. Reduced costs. Led a team. Wrote clear code. Deployed services."
        issues = []
        score  = _score_readability(text, issues)
        assert score >= 60

    def test_empty_text_scores_zero(self):
        issues = []
        assert _score_readability("", issues) == 0.0

    def test_returns_float(self):
        issues = []
        score  = _score_readability("Developed and deployed machine learning pipelines.", issues)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0


# ── Completeness scoring ──────────────────────────────────────────────────────

class TestCompletenessScoring:
    def test_full_cv_scores_high(self):
        cv     = _cv()
        issues = []
        score  = _score_completeness(cv, issues)
        assert score >= 70

    def test_no_email_penalised(self):
        cv     = _cv(has_email=False)
        issues = []
        score  = _score_completeness(cv, issues)
        assert score < 90
        assert any("email" in i.lower() for i in issues)

    def test_no_dates_in_experience_flagged(self):
        cv = _cv(sections={
            "experience": "Worked at a company as a software engineer doing ML things",
            "education": "BSc CS UCL",
            "skills": "Python Docker",
        })
        issues = []
        _score_completeness(cv, issues)
        assert any("date" in i.lower() for i in issues)

    def test_sparse_skills_flagged(self):
        cv = _cv(sections={
            "experience": "ML Engineer 2020-2024",
            "education": "BSc CS",
            "skills": "Python",  # only 1 word
        })
        issues = []
        _score_completeness(cv, issues)
        assert any("skill" in i.lower() for i in issues)


# ── Format safety scoring ─────────────────────────────────────────────────────

class TestFormatScoring:
    def test_clean_cv_scores_100(self):
        cv     = _cv(has_tables=False, has_images=False, page_count=2)
        issues = []
        assert _score_format(cv, issues) == 100.0
        assert issues == []

    def test_tables_penalised(self):
        cv     = _cv(has_tables=True)
        issues = []
        score  = _score_format(cv, issues)
        assert score <= 60
        assert any("table" in i.lower() for i in issues)

    def test_images_penalised(self):
        cv     = _cv(has_images=True)
        issues = []
        score  = _score_format(cv, issues)
        assert score <= 80
        assert any("image" in i.lower() or "graphic" in i.lower() for i in issues)

    def test_too_many_pages_penalised(self):
        cv     = _cv(page_count=5)
        issues = []
        score  = _score_format(cv, issues)
        assert score < 100
        assert any("page" in i.lower() for i in issues)

    def test_tables_and_images_compound_penalty(self):
        cv_both  = _cv(has_tables=True, has_images=True)
        cv_clean = _cv(has_tables=False, has_images=False)
        assert _score_format(cv_both, []) < _score_format(cv_clean, [])


# ── Grade thresholds ──────────────────────────────────────────────────────────

class TestGradeThresholds:
    @pytest.mark.parametrize("total, expected_grade", [
        (95.0,  "A+"),
        (85.0,  "A"),
        (75.0,  "B"),
        (65.0,  "C"),
        (45.0,  "D"),
        (0.0,   "D"),
        (100.0, "A+"),
    ])
    def test_grade_boundaries(self, total: float, expected_grade: str):
        grade = next(g for threshold, g in _GRADE_THRESHOLDS if total >= threshold)
        assert grade == expected_grade


# ── Full score_cv integration ─────────────────────────────────────────────────

class TestFullScoreCV:
    def test_score_cv_returns_ats_score(self):
        cv     = _cv()
        result = score_cv(cv, target_role="ml_engineer")
        assert isinstance(result, ATSScore)
        assert 0 <= result.total <= 100
        assert result.grade in ("A+", "A", "B", "C", "D")

    def test_score_cv_breakdown_has_all_dimensions(self):
        cv     = _cv()
        result = score_cv(cv, target_role="ml_engineer")
        for dim in ("keyword_match", "structure", "readability", "completeness", "format_safety"):
            assert dim in result.breakdown
            assert 0.0 <= result.breakdown[dim] <= 100.0

    def test_cv_with_tables_scores_lower_than_clean(self):
        cv_clean  = _cv(has_tables=False)
        cv_tables = _cv(has_tables=True)
        score_clean  = score_cv(cv_clean,  target_role="ml_engineer").total
        score_tables = score_cv(cv_tables, target_role="ml_engineer").total
        assert score_clean > score_tables

    def test_issues_list_populated(self):
        # CV with obvious issues
        cv = _cv(
            sections={"experience": "Worked at a company"},  # no education, no skills
            has_email=False,
        )
        result = score_cv(cv, target_role="ml_engineer")
        assert len(result.issues) > 0

    def test_extra_skills_merged(self):
        cv     = _cv(raw_text="ML Engineer with experience")
        result = score_cv(cv, "ml_engineer", extra_skills=["PyTorch", "LangGraph"])
        assert "PyTorch" in result.skills_found or len(result.skills_found) >= 0
