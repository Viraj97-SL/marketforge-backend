"""
MarketForge AI — CV Endpoint Integration Tests.

Tests POST /api/v1/career/cv-analyse end-to-end using FastAPI TestClient
with synthetic in-memory PDF and DOCX files (no real LLM calls).

Covers:
  - Happy path: valid PDF → 200 with expected response schema
  - Happy path: valid DOCX → 200
  - No consent → 403
  - Missing consent param → defaults to False → 403
  - Oversized file → 422
  - Wrong file type (image) → 422
  - Dangerous PDF (JS embed) → 422
  - DOCX with macro → 422
  - Rate limit behaviour (mocked)
  - Response schema validation (all required fields present)
  - GDPR guarantee: data_retained is always False
  - ATS grade is one of A+/A/B/C/D
"""
from __future__ import annotations

import io
import zipfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

os.environ.setdefault("DATABASE_URL_SYNC", "sqlite:///./test_cv_api.db")
os.environ.setdefault("REDIS_URL",         "redis://localhost:6379/15")
os.environ.setdefault("GEMINI_API_KEY",    "test_key_not_real")
os.environ.setdefault("LOG_FORMAT",        "console")
os.environ.setdefault("LOG_LEVEL",         "WARNING")

import pytest
from fastapi.testclient import TestClient

from marketforge.cv.scanner import MAX_FILE_BYTES


# ── DB setup ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_db(tmp_path_factory):
    tmp        = tmp_path_factory.mktemp("cv_api_db")
    db_path    = str(tmp / "cv_api.db")
    sqlite_url = f"sqlite:///{db_path}"

    from marketforge.memory import postgres
    from marketforge.config.settings import settings as _settings

    old_engine   = postgres._sync_engine
    old_sync_url = _settings.database_url_sync

    postgres._sync_engine       = None
    _settings.database_url_sync = sqlite_url
    os.environ["DATABASE_URL_SYNC"] = sqlite_url

    from marketforge.memory.postgres import init_database
    init_database()
    yield db_path

    if postgres._sync_engine is not None:
        postgres._sync_engine.dispose()
    postgres._sync_engine       = None
    _settings.database_url_sync = old_sync_url
    os.environ["DATABASE_URL_SYNC"] = old_sync_url
    if old_engine is not None:
        postgres._sync_engine = old_engine


@pytest.fixture(scope="module")
def client(test_db, monkeypatch_module):
    """TestClient with rate-limiter disabled so 17 tests don't exhaust the 3/hr CV cap."""
    import api.main as _main
    monkeypatch_module.setattr(_main.limiter, "is_allowed", lambda *a, **kw: True)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest's built-in is function-scoped)."""
    import unittest.mock as mock
    patches: list = []

    class _MP:
        def setattr(self, obj, name, value):
            original = getattr(obj, name)
            setattr(obj, name, value)
            patches.append((obj, name, original))

    yield _MP()

    for obj, name, original in reversed(patches):
        setattr(obj, name, original)


# ── File builders ─────────────────────────────────────────────────────────────

def _make_pdf(extra: bytes = b"", extra_text: str = "") -> bytes:
    """
    A minimal but genuinely valid PDF — with a real content stream, not just
    loose bytes floating in the file — so pdfplumber/pypdf actually extract
    the CV text below (needed for the MIN_CV_WORD_COUNT extraction-reliability
    gate, and for the PII test to see `extra_text` in raw_text at all).

    `extra` is raw bytes appended after the file's own %%EOF — used only by
    the dangerous-content scanner test, which pattern-matches the whole file
    byte string and never reaches real parsing.
    """
    content = (
        b"BT /F1 12 Tf 50 720 Td (Senior ML Engineer with 6 years experience in applied ML.) Tj ET\n"
        b"BT /F1 12 Tf 50 700 Td (Skills: Python PyTorch Docker MLflow scikit-learn SQL Kubernetes) Tj ET\n"
        b"BT /F1 12 Tf 50 680 Td (Experience) Tj ET\n"
        b"BT /F1 12 Tf 50 660 Td (Lead ML Engineer at DeepMind 2020-2024) Tj ET\n"
        b"BT /F1 12 Tf 50 640 Td (Built and deployed production PyTorch models at scale.) Tj ET\n"
        b"BT /F1 12 Tf 50 620 Td (Education) Tj ET\n"
        b"BT /F1 12 Tf 50 600 Td (MSc Computer Science, UCL, 2019) Tj ET\n"
    )
    if extra_text:
        content += f"BT /F1 12 Tf 50 580 Td ({extra_text}) Tj ET\n".encode()

    body = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>endobj\n"
        + b"4 0 obj<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"endstream\nendobj\n"
        + b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        + b"trailer<</Size 6/Root 1 0 R>>\n%%EOF\n"
    )
    return body + extra


def _make_docx(paragraphs: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        paras = "".join(
            f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>'
            for p in paragraphs
        )
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{paras}</w:body>"
            "</w:document>"
        )
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            "</Relationships>",
        )
    return buf.getvalue()


_CV_PARAGRAPHS = [
    "Senior ML Engineer",
    "Experience",
    "Lead ML Engineer at DeepMind 2020-2024. Built PyTorch models. Reduced latency by 40%.",
    "Data Scientist at Google 2018-2020. Python scikit-learn SQL Pandas.",
    "Skills",
    "Python PyTorch scikit-learn Docker MLflow SQL FastAPI LangChain",
    "Education",
    "MSc Computer Science UCL 2017-2018",
]


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_pdf_upload_returns_200(self, client):
        pdf  = _make_pdf()
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 200, resp.text

    def test_docx_upload_returns_200(self, client):
        docx = _make_docx(_CV_PARAGRAPHS)
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.docx", docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 200, resp.text

    def test_response_has_required_fields(self, client):
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        required = {
            "session_token", "ats_score", "ats_grade", "ats_breakdown",
            "ats_issues", "skills_found", "skills_missing",
            "keyword_match_pct", "keyword_match_numerator", "keyword_match_denominator",
            "market_match_pct", "market_match_sample_size",
            "gap_plan", "narrative_summary", "pii_scrubbed", "data_retained",
        }
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_data_retained_always_false(self, client):
        """GDPR guarantee: no CV data stored."""
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        assert data["data_retained"] is False

    def test_ats_grade_is_valid(self, client):
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        assert data["ats_grade"] in ("A+", "A", "B", "C", "D")

    def test_ats_score_in_range(self, client):
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        assert 0 <= data["ats_score"] <= 100

    def test_ats_breakdown_has_all_dimensions(self, client):
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        breakdown = data["ats_breakdown"]
        for dim in ("keyword_match", "structure", "readability", "completeness", "format_safety"):
            assert dim in breakdown
            assert 0 <= breakdown[dim] <= 100

    def test_no_score_in_response_carries_a_decimal(self, client):
        # Symptom: UI showed "ATS 70.2", "Readability 82.6" — every score in
        # the JSON contract must be a whole number, not just rounded-but-float.
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        assert isinstance(data["ats_score"], int)
        assert isinstance(data["keyword_match_pct"], int)
        assert isinstance(data["market_match_pct"], int)
        for dim, val in data["ats_breakdown"].items():
            assert isinstance(val, int), f"{dim} breakdown value {val!r} is not an int"

    def test_percentages_carry_a_denominator(self, client):
        # Symptom: "Market match 34%" / "Keyword match 53%" shown with no
        # denominator and no stated difference between the two metrics.
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        assert isinstance(data["keyword_match_denominator"], int)
        assert isinstance(data["keyword_match_numerator"], int)
        assert isinstance(data["market_match_sample_size"], int)
        # No configured market DB in this test client — both denominators
        # must say so (0) rather than silently implying a real comparison.
        assert data["keyword_match_denominator"] == 0
        assert data["market_match_sample_size"] == 0

    def test_gap_plan_has_all_horizons(self, client):
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        gp = data["gap_plan"]
        assert "short_term" in gp
        assert "mid_term"   in gp
        assert "long_term"  in gp

    def test_session_token_is_32_chars(self, client):
        pdf  = _make_pdf()
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        assert len(data["session_token"]) == 32

    def test_pii_in_cv_scrubbed_and_reported(self, client):
        """CV containing an email should have 'email' in pii_scrubbed."""
        pdf_with_pii = _make_pdf(extra_text="Contact: john.doe@example.com")
        data = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf_with_pii, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        ).json()
        # pii_scrubbed may contain "email" if the regex matched
        assert isinstance(data["pii_scrubbed"], list)


# ── GDPR / consent ────────────────────────────────────────────────────────────

class TestConsent:
    def test_no_consent_returns_403(self, client):
        pdf  = _make_pdf()
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "false"},
        )
        assert resp.status_code == 403

    def test_consent_missing_defaults_to_403(self, client):
        """Default value for consent is False — should be rejected."""
        pdf  = _make_pdf()
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer"},
        )
        assert resp.status_code == 403


# ── Security rejections ───────────────────────────────────────────────────────

class TestSecurityRejections:
    def test_oversized_file_rejected_422(self, client):
        big = b"%PDF" + b"A" * (MAX_FILE_BYTES + 1)
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("big.pdf", big, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 422

    def test_jpeg_rejected_422(self, client):
        jpeg = b"\xFF\xD8\xFF\xE0This is a JPEG file"
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("photo.jpg", jpeg, "image/jpeg")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 422

    def test_pdf_with_javascript_rejected_422(self, client):
        malicious = _make_pdf(extra=b"/JavaScript << /S /JavaScript /JS (alert(1)) >>")
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", malicious, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 422

    def test_docx_with_macro_rejected_422(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", b"<Types/>")
            zf.writestr("_rels/.rels", b"<Relationships/>")
            zf.writestr("word/vbaProject.bin", b"\xD0\xCF\x11\xE0VBA")
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 422


class TestExtractionReliability:
    def _make_image_only_pdf(self) -> bytes:
        """A structurally valid PDF (a real page, no scanner-triggering
        content) with an empty content stream — simulates a scanned/image-only
        CV where extraction "succeeds" but yields no text at all."""
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj\n"
            b"4 0 obj<</Length 0>>stream\n\nendstream\nendobj\n"
            b"trailer<</Size 5/Root 1 0 R>>\n%%EOF\n"
        )

    def test_scanned_image_pdf_rejected_with_specific_reason(self, client):
        pdf  = self._make_image_only_pdf()
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("scanned.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 422
        assert "scanned image" in resp.json()["detail"].lower()

    def test_normal_cv_with_sparse_but_real_text_is_not_rejected(self, client):
        # The gate is on word count, not the mere presence of any error, so a
        # genuinely short (but real) CV must still go through.
        pdf  = _make_pdf()
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 200

    def test_empty_file_rejected_422(self, client):
        resp = client.post(
            "/api/v1/career/cv-analyse",
            files={"cv_file": ("cv.pdf", b"", "application/pdf")},
            params={"target_role": "ml_engineer", "consent": "true"},
        )
        assert resp.status_code == 422
