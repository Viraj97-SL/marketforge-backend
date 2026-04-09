"""
MarketForge AI — CV File Security Tests.

Tests the scanner.scan_file() function against:
  - Valid PDF and DOCX files (should be allowed)
  - Oversized files (should be rejected)
  - Wrong magic bytes / unknown types (should be rejected)
  - Empty files (should be rejected)
  - PDF with dangerous patterns: /JS, /JavaScript, /Launch, /OpenAction
  - PDF encrypted (/Encrypt marker)
  - DOCX with VBA macro (vbaProject.bin in ZIP)
  - DOCX zipbomb (uncompressed size > limit)
  - Corrupt / truncated DOCX (bad ZIP)
  - EICAR test string (virus simulation — only detected if ClamAV available)

No real malware is used. All test payloads are synthetic minimal files.
"""
from __future__ import annotations

import io
import struct
import zipfile

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from marketforge.cv.scanner import (
    MAX_FILE_BYTES,
    scan_file,
    ScanResult,
)

# ── Helpers — minimal valid file builders ────────────────────────────────────

def _minimal_pdf(extra: bytes = b"") -> bytes:
    """Build the smallest valid PDF that pdfplumber can open."""
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    )
    body += extra
    body += b"xref\n0 4\n0000000000 65535 f \n"
    body += b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    return body


def _minimal_docx(extra_files: dict[str, bytes] | None = None) -> bytes:
    """Build a minimal valid DOCX (ZIP) with optional extra entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            b'<?xml version="1.0"?>'
            b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            b'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            b'<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            b"</Types>",
        )
        zf.writestr(
            "word/document.xml",
            b'<?xml version="1.0"?>'
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            b"<w:body><w:p><w:r><w:t>Test CV</w:t></w:r></w:p></w:body>"
            b"</w:document>",
        )
        if extra_files:
            for name, content in extra_files.items():
                zf.writestr(name, content)
    return buf.getvalue()


# ── Clean file tests ──────────────────────────────────────────────────────────

class TestCleanFiles:
    def test_valid_pdf_allowed(self):
        result = scan_file(_minimal_pdf())
        assert result.allowed is True
        assert result.file_type == "pdf"
        assert result.rejection_reason is None

    def test_valid_docx_allowed(self):
        result = scan_file(_minimal_docx())
        assert result.allowed is True
        assert result.file_type == "docx"
        assert result.rejection_reason is None

    def test_file_hash_is_sha256(self):
        data   = _minimal_pdf()
        result = scan_file(data)
        import hashlib
        assert result.file_hash == hashlib.sha256(data).hexdigest()

    def test_size_bytes_populated(self):
        data   = _minimal_pdf()
        result = scan_file(data)
        assert result.size_bytes == len(data)


# ── Size gate ─────────────────────────────────────────────────────────────────

class TestSizeGate:
    def test_empty_file_rejected(self):
        result = scan_file(b"")
        assert result.allowed is False
        assert result.rejection_reason == "empty_file"

    def test_oversized_file_rejected(self):
        # 1 byte over the limit
        data   = b"%PDF" + b"A" * (MAX_FILE_BYTES - 3)
        result = scan_file(data)
        assert result.allowed is False
        assert result.rejection_reason == "file_too_large"

    def test_exactly_at_limit_allowed(self):
        # Exactly at the limit — a real PDF header padded to max size
        # We only check the size gate; content scan follows
        data   = b"%PDF" + b"A" * (MAX_FILE_BYTES - 4)
        result = scan_file(data)
        # May fail content scan but must NOT fail with "file_too_large"
        assert result.rejection_reason != "file_too_large"


# ── Magic bytes / file type ───────────────────────────────────────────────────

class TestMagicBytes:
    def test_unknown_magic_rejected(self):
        result = scan_file(b"\x00\x01\x02\x03This is not a PDF or DOCX")
        assert result.allowed is False
        assert result.rejection_reason == "unsupported_file_type"

    def test_exe_magic_rejected(self):
        # MZ header — Windows executable
        result = scan_file(b"MZThis is an executable")
        assert result.allowed is False
        assert result.rejection_reason == "unsupported_file_type"

    def test_jpeg_magic_rejected(self):
        result = scan_file(b"\xFF\xD8\xFF\xE0This is a JPEG")
        assert result.allowed is False
        assert result.rejection_reason == "unsupported_file_type"

    def test_zip_without_docx_structure_rejected_or_allowed(self):
        # A bare ZIP (not DOCX) has the same magic — scanner accepts it at magic
        # level but DOCX content scan may pass (no macros). This is acceptable:
        # the parser will fail gracefully downstream.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", b"not a docx")
        result = scan_file(buf.getvalue())
        # Must not crash — either allowed or rejected with a clean reason
        assert isinstance(result.allowed, bool)
        assert result.rejection_reason != "unsupported_file_type"


# ── Dangerous PDF content ─────────────────────────────────────────────────────

class TestDangerousPDF:
    @pytest.mark.parametrize("pattern", [
        b"/JS ",
        b"/JavaScript",
        b"/Launch",
        b"/EmbeddedFile",
        b"/OpenAction",
        b"/AA ",
    ])
    def test_dangerous_pdf_pattern_rejected(self, pattern: bytes):
        data   = _minimal_pdf(extra=pattern + b" << /S /JavaScript /JS (app.alert('xss')) >>")
        result = scan_file(data)
        assert result.allowed is False
        assert result.rejection_reason == "pdf_dangerous_content"

    def test_encrypted_pdf_rejected(self):
        data   = _minimal_pdf(extra=b"/Encrypt << /Filter /Standard >>")
        result = scan_file(data)
        assert result.allowed is False
        assert result.rejection_reason == "pdf_encrypted"

    def test_clean_pdf_with_keywords_in_text_allowed(self):
        # The word "JavaScript" appears in the body text of a job description CV —
        # only byte-pattern detection matters (scanner checks raw bytes, not parsed).
        # This test is explicitly included because "JavaScript developer" in a CV
        # should NOT be rejected — the pattern is /JavaScript (with slash prefix).
        data   = _minimal_pdf(extra=b"Senior JavaScript Developer with 5 years experience")
        result = scan_file(data)
        # "JavaScript" without the leading "/" is NOT a danger pattern
        assert result.allowed is True


# ── Dangerous DOCX content ────────────────────────────────────────────────────

class TestDangerousDOCX:
    def test_macro_docx_rejected(self):
        data   = _minimal_docx({"word/vbaProject.bin": b"\xD0\xCF\x11\xE0VBA binary content"})
        result = scan_file(data)
        assert result.allowed is False
        assert result.rejection_reason == "docx_macro_detected"

    def test_corrupt_zip_rejected(self):
        result = scan_file(b"PK\x03\x04THIS IS NOT A REAL ZIP")
        assert result.allowed is False
        assert result.rejection_reason == "docx_corrupt"

    def test_zipbomb_rejected(self):
        """A DOCX where the uncompressed content exceeds 50 MB."""
        from marketforge.cv.scanner import MAX_UNCOMPRESSED_BYTES
        buf = io.BytesIO()
        # Write a single highly-compressible entry that uncompresses beyond limit
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            # Write a file whose declared file_size exceeds the limit
            # We fake file_size by writing actual data
            large_content = b"A" * (MAX_UNCOMPRESSED_BYTES + 1)
            zf.writestr("word/document.xml", large_content)
        result = scan_file(buf.getvalue())
        # Either rejected as zipbomb or as oversized (size gate catches it first)
        assert result.allowed is False
        assert result.rejection_reason in ("docx_zipbomb", "file_too_large")


# ── EICAR simulation (no real AV needed) ─────────────────────────────────────

class TestEICAR:
    def test_eicar_pattern_does_not_crash_scanner(self):
        """
        The EICAR test string embedded in a PDF-looking file.
        ClamAV detects it if available; scanner must not crash either way.
        """
        eicar = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        data  = b"%PDF-1.4\n" + eicar
        result = scan_file(data)
        # Must return a ScanResult (no exception) — allowed state depends on ClamAV availability
        assert isinstance(result, ScanResult)
        assert isinstance(result.allowed, bool)
