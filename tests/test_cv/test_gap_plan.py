"""
MarketForge AI — CV Gap Plan Generation Tests.

Regression tests for the "generic plan, no pipeline figures, uneven bullet
counts" bug: the plan's structure (bullet count, which skills are named) must
come entirely from the deterministic ML-ranked buckets computed by
gap_analyser, and the model's narrative must be discarded in favour of a
deterministic fallback if it names any skill outside what the pipeline
computed, or if the call fails outright.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

os.environ.setdefault("DATABASE_URL_SYNC", "sqlite:///./test_gap_plan.db")
os.environ.setdefault("REDIS_URL",         "redis://localhost:6379/15")
os.environ.setdefault("GEMINI_API_KEY",    "test_key_not_real")
os.environ.setdefault("LOG_FORMAT",        "console")
os.environ.setdefault("LOG_LEVEL",         "WARNING")

import pytest

import api.main as _main


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _fake_llm(content: str):
    class _FakeLLM:
        def __init__(self, *a, **kw):
            pass

        def invoke(self, messages):
            return _FakeResponse(content)

    return _FakeLLM


class TestEmptyBucketsSkipTheLLM:
    @pytest.mark.asyncio
    async def test_no_llm_call_when_all_buckets_empty(self, monkeypatch):
        called = {"n": 0}

        class _ShouldNotBeCalled:
            def __init__(self, *a, **kw):
                called["n"] += 1

        monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _ShouldNotBeCalled)

        plan, narrative = await _main._generate_cv_gap_plan(
            ats_score=70, skills_found=["Python"], ml_short_term=[], ml_mid_term=[],
            ml_long_term=[], target_role="ml_engineer", match_pct=40,
        )
        assert called["n"] == 0
        assert "isn't yet enough" in narrative


class TestDeterministicBulletStructure:
    @pytest.mark.asyncio
    async def test_bullet_count_matches_ml_buckets_exactly(self, monkeypatch):
        # Bug: one horizon got 1 bullet, another got 4, regardless of how
        # many skills were actually ranked into each. Bullet count must now
        # equal the ML bucket size, by construction.
        monkeypatch.setattr(
            "langchain_google_genai.ChatGoogleGenerativeAI",
            _fake_llm("Solid start; closing these gaps will raise your market fit."),
        )
        plan, _ = await _main._generate_cv_gap_plan(
            ats_score=70, skills_found=["Python"],
            ml_short_term=["Docker", "Kubernetes"],
            ml_mid_term=["LangGraph"],
            ml_long_term=["PyTorch", "CUDA", "Rust"],
            target_role="ml_engineer", match_pct=40,
        )
        assert len(plan.short_term) == 2
        assert len(plan.mid_term) == 1
        assert len(plan.long_term) == 3
        assert any("Docker" in b for b in plan.short_term)
        assert any("LangGraph" in b for b in plan.mid_term)
        assert any("Rust" in b for b in plan.long_term)


class TestNarrativeGuardrail:
    @pytest.mark.asyncio
    async def test_narrative_naming_extra_skill_is_rejected(self, monkeypatch):
        # Model invents "TensorFlow" though it was never in the CV or gap list.
        monkeypatch.setattr(
            "langchain_google_genai.ChatGoogleGenerativeAI",
            _fake_llm("You should also learn TensorFlow to strengthen your CV."),
        )
        plan, narrative = await _main._generate_cv_gap_plan(
            ats_score=70, skills_found=["Python"], ml_short_term=["Docker"],
            ml_mid_term=[], ml_long_term=[], target_role="ml_engineer", match_pct=40,
        )
        assert "TensorFlow" not in narrative
        assert "70" in narrative   # fell back to the deterministic template

    @pytest.mark.asyncio
    async def test_clean_narrative_is_used_as_is(self, monkeypatch):
        monkeypatch.setattr(
            "langchain_google_genai.ChatGoogleGenerativeAI",
            _fake_llm("Your Docker and Python background is a solid foundation for this role."),
        )
        plan, narrative = await _main._generate_cv_gap_plan(
            ats_score=70, skills_found=["Python"], ml_short_term=["Docker"],
            ml_mid_term=[], ml_long_term=[], target_role="ml_engineer", match_pct=40,
        )
        assert "solid foundation" in narrative

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_deterministic_plan_and_narrative(self, monkeypatch):
        class _BrokenLLM:
            def __init__(self, *a, **kw):
                raise RuntimeError("no api key")

        monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _BrokenLLM)
        plan, narrative = await _main._generate_cv_gap_plan(
            ats_score=70, skills_found=["Python"], ml_short_term=["Docker"],
            ml_mid_term=[], ml_long_term=[], target_role="ml_engineer", match_pct=40,
        )
        assert plan.short_term == ["Complete a course or certification in Docker"]
        assert "70" in narrative
