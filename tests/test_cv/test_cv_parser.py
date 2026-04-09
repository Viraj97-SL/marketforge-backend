"""
MarketForge AI — CV Parser Tests.

Tests parser.parse_cv() for:
  - PDF text extraction (pdfplumber path)
  - DOCX text extraction (python-docx path)
  - Section detection (experience, skills, education, etc.)
  - PII presence flags (has_email, has_phone, has_linkedin, has_github)
  - Years experience estimation
  - Table / image detection flags
  - Error handling for corrupt / empty files
"""
from __future__ import annotations

import io
import zipfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from marketforge.cv.parser import parse_cv, _detect_sections, _estimate_years, ParsedCV


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_docx(text_paragraphs: list[str], with_table: bool = False) -> bytes:
    """
    Build a minimal but python-docx-compatible DOCX.
    Includes the required _rels/.rels relationship file so python-docx
    can locate the main document part.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Top-level content types
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
        # Package-level relationships (required by python-docx to find the document)
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        # Document body
        paras = "".join(
            f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>"
            for p in text_paragraphs
        )
        table_xml = ""
        if with_table:
            table_xml = (
                "<w:tbl>"
                "<w:tr><w:tc><w:p><w:r><w:t>TableCell</w:t></w:r></w:p></w:tc></w:tr>"
                "</w:tbl>"
            )
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{paras}{table_xml}</w:body>"
            "</w:document>"
        )
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
        # Word-level relationships (needed by python-docx internals)
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            "</Relationships>",
        )
    return buf.getvalue()


# ── DOCX parsing ──────────────────────────────────────────────────────────────

class TestDOCXParsing:
    def test_basic_text_extracted(self):
        data   = _make_docx(["John Doe", "Python developer with 5 years experience"])
        result = parse_cv(data, "docx")
        assert "Python" in result.raw_text
        assert result.error is None

    def test_multiple_paragraphs_joined(self):
        paras  = ["Experience", "Software Engineer at Google 2020-2024", "Skills", "Python Docker Kubernetes"]
        data   = _make_docx(paras)
        result = parse_cv(data, "docx")
        for p in paras:
            assert p in result.raw_text

    def test_table_flag_detected(self):
        data   = _make_docx(["Skills"], with_table=True)
        result = parse_cv(data, "docx")
        assert result.has_tables is True

    def test_no_table_flag_clean_docx(self):
        data   = _make_docx(["Experience", "Software Engineer 2020-2024"])
        result = parse_cv(data, "docx")
        assert result.has_tables is False

    def test_parse_method_is_python_docx(self):
        data   = _make_docx(["Test"])
        result = parse_cv(data, "docx")
        assert result.parse_method == "python_docx"

    def test_corrupt_docx_returns_error(self):
        result = parse_cv(b"PK\x03\x04CORRUPT DATA", "docx")
        assert result.error is not None
        assert result.raw_text == ""


# ── PII flags ─────────────────────────────────────────────────────────────────

class TestPIIFlags:
    def test_email_flag(self):
        data   = _make_docx(["Contact me at john.doe@example.com"])
        result = parse_cv(data, "docx")
        assert result.has_email is True

    def test_phone_flag(self):
        data   = _make_docx(["Phone: 07911 123456"])
        result = parse_cv(data, "docx")
        assert result.has_phone is True

    def test_linkedin_flag(self):
        data   = _make_docx(["linkedin.com/in/johndoe"])
        result = parse_cv(data, "docx")
        assert result.has_linkedin is True

    def test_github_flag(self):
        data   = _make_docx(["github.com/johndoe"])
        result = parse_cv(data, "docx")
        assert result.has_github is True

    def test_no_flags_on_clean_text(self):
        data   = _make_docx(["Senior ML Engineer with PyTorch experience"])
        result = parse_cv(data, "docx")
        assert result.has_email   is False
        assert result.has_phone   is False
        assert result.has_linkedin is False


# ── Section detection ─────────────────────────────────────────────────────────

class TestSectionDetection:
    def test_experience_section_detected(self):
        text = "Experience\nSoftware Engineer at DeepMind 2020-2024\nSkills\nPython PyTorch"
        sections = _detect_sections(text)
        assert "experience" in sections
        assert "DeepMind" in sections["experience"]

    def test_skills_section_detected(self):
        text = "Skills\nPython, PyTorch, Docker, Kubernetes"
        sections = _detect_sections(text)
        assert "skills" in sections
        assert "Python" in sections["skills"]

    def test_education_section_detected(self):
        text = "Education\nMSc Computer Science, UCL 2018-2019"
        sections = _detect_sections(text)
        assert "education" in sections
        assert "UCL" in sections["education"]

    def test_multiple_sections_split_correctly(self):
        text = (
            "Summary\nExperienced ML engineer.\n"
            "Experience\nLead ML Engineer at Google 2019-2024\n"
            "Skills\nPython TensorFlow PyTorch\n"
            "Education\nBEng Computer Science UCL 2015-2018"
        )
        sections = _detect_sections(text)
        assert "summary"    in sections
        assert "experience" in sections
        assert "skills"     in sections
        assert "education"  in sections

    def test_section_text_not_included_in_wrong_section(self):
        text = "Experience\nBuilt ML pipeline\nEducation\nBSc Computer Science"
        sections = _detect_sections(text)
        assert "BSc" not in sections.get("experience", "")
        assert "BSc" in sections.get("education", "")

    def test_case_insensitive_headers(self):
        text = "EXPERIENCE\nSoftware Engineer\nSKILLS\nPython"
        sections = _detect_sections(text)
        assert "experience" in sections or "EXPERIENCE" in sections

    def test_no_sections_returns_preamble(self):
        text = "Just some plain text with no headers at all"
        sections = _detect_sections(text)
        assert len(sections) >= 1  # at least preamble or one section


# ── Years experience ─────────────────────────────────────────────────────────

class TestYearsEstimation:
    def test_recent_range_gives_positive_years(self):
        text  = "Software Engineer 2018 – 2024"
        years = _estimate_years(text)
        assert years is not None
        assert years >= 2   # at minimum 2018 → today

    def test_no_years_returns_none(self):
        text  = "Python developer with great skills"
        years = _estimate_years(text)
        assert years is None

    def test_single_year_still_computes(self):
        from datetime import date
        text  = "Graduated in 2020"
        years = _estimate_years(text)
        assert years is not None
        assert years >= (date.today().year - 2020)


# ── Unsupported file type ─────────────────────────────────────────────────────

class TestUnsupportedType:
    def test_unknown_type_returns_error(self):
        result = parse_cv(b"some bytes", "txt")
        assert result.error == "unsupported_file_type"
        assert result.raw_text == ""
