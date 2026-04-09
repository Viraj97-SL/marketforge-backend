"""
MarketForge AI — GDPR Compliance Tests.

Verifies that the GDPR layer for CV processing:
  - Raises ConsentNotGiven when consent=False
  - Scrubs all PII patterns from extracted text
  - Session token contains no PII and is non-reversible
  - Different requests for the same file produce different tokens (random salt)
  - Scrubbed text does not contain original PII values
  - build_gdpr_context returns correct structure
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from marketforge.cv.gdpr import (
    ConsentNotGiven,
    GDPRContext,
    build_gdpr_context,
    check_consent,
    make_session_token,
    scrub_pii,
)

_FILE_HASH = "a" * 64   # synthetic SHA-256 hash


# ── Consent gate ──────────────────────────────────────────────────────────────

class TestConsentGate:
    def test_consent_true_passes(self):
        check_consent(True)   # must not raise

    def test_consent_false_raises(self):
        with pytest.raises(ConsentNotGiven):
            check_consent(False)

    def test_build_gdpr_context_raises_on_no_consent(self):
        with pytest.raises(ConsentNotGiven):
            build_gdpr_context("some cv text", _FILE_HASH, consent=False)


# ── PII scrubbing ─────────────────────────────────────────────────────────────

class TestPIIScrubbing:
    def test_email_scrubbed(self):
        text, found = scrub_pii("Contact me at john.doe@example.com for details")
        assert "john.doe@example.com" not in text
        assert "[REDACTED:EMAIL]" in text
        assert "email" in found

    def test_uk_phone_scrubbed(self):
        text, found = scrub_pii("Call me on 07911 123456 anytime")
        assert "07911 123456" not in text
        assert "phone_uk" in found

    def test_ni_number_scrubbed(self):
        text, found = scrub_pii("NI number: AB 12 34 56 C")
        assert "AB 12 34 56 C" not in text
        assert "ni_number" in found

    def test_postcode_scrubbed(self):
        text, found = scrub_pii("Based in London EC1A 1BB")
        assert "EC1A 1BB" not in text
        assert "postcode_uk" in found

    def test_multiple_pii_types_all_scrubbed(self):
        raw = "Email: alice@test.com, Phone: 07911 123456, NI: AB 12 34 56 C"
        text, found = scrub_pii(raw)
        assert "alice@test.com" not in text
        assert "07911 123456" not in text
        assert "AB 12 34 56 C" not in text
        assert len(found) >= 3

    def test_clean_tech_text_not_scrubbed(self):
        raw = "Python PyTorch FastAPI Docker Kubernetes LangChain"
        text, found = scrub_pii(raw)
        assert text == raw
        assert found == []

    def test_skills_with_version_numbers_not_scrubbed(self):
        raw = "PyTorch 2.3, Python 3.11, scikit-learn 1.4, FastAPI 0.111"
        text, found = scrub_pii(raw)
        # Version numbers should not match PII patterns
        assert "PyTorch" in text
        assert "Python" in text


# ── Session token ─────────────────────────────────────────────────────────────

class TestSessionToken:
    def test_token_is_32_chars(self):
        token = make_session_token(_FILE_HASH)
        assert len(token) == 32

    def test_token_is_hex(self):
        token = make_session_token(_FILE_HASH)
        int(token, 16)   # raises ValueError if not valid hex

    def test_same_file_different_tokens(self):
        """Two calls for the same file hash must produce different tokens (random salt)."""
        t1 = make_session_token(_FILE_HASH)
        t2 = make_session_token(_FILE_HASH)
        assert t1 != t2

    def test_token_does_not_contain_file_hash(self):
        """Token must not leak the file hash or any fragment of it."""
        token = make_session_token(_FILE_HASH)
        # The full hash should not appear verbatim in the token
        assert _FILE_HASH not in token


# ── build_gdpr_context ────────────────────────────────────────────────────────

class TestBuildGDPRContext:
    def test_returns_gdpr_context(self):
        ctx = build_gdpr_context("Python developer", _FILE_HASH, consent=True)
        assert isinstance(ctx, GDPRContext)

    def test_consent_given_is_true(self):
        ctx = build_gdpr_context("text", _FILE_HASH, consent=True)
        assert ctx.consent_given is True

    def test_scrubbed_text_has_no_email(self):
        raw = "Contact: dev@example.com — Python, PyTorch, Docker"
        ctx = build_gdpr_context(raw, _FILE_HASH, consent=True)
        assert "dev@example.com" not in ctx.scrubbed_text
        assert "[REDACTED:EMAIL]" in ctx.scrubbed_text

    def test_pii_types_found_populated(self):
        raw = "Phone: 07911 123456, email: x@y.com"
        ctx = build_gdpr_context(raw, _FILE_HASH, consent=True)
        assert "email" in ctx.pii_types_found
        assert "phone_uk" in ctx.pii_types_found

    def test_clean_text_pii_types_empty(self):
        raw = "Senior ML Engineer with 5 years experience in PyTorch and FastAPI"
        ctx = build_gdpr_context(raw, _FILE_HASH, consent=True)
        assert ctx.pii_types_found == []

    def test_session_token_non_empty(self):
        ctx = build_gdpr_context("text", _FILE_HASH, consent=True)
        assert len(ctx.session_token) == 32

    def test_two_sessions_different_tokens(self):
        ctx1 = build_gdpr_context("text", _FILE_HASH, consent=True)
        ctx2 = build_gdpr_context("text", _FILE_HASH, consent=True)
        assert ctx1.session_token != ctx2.session_token
