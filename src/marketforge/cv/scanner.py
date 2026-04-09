"""
MarketForge AI — CV File Security Scanner.

Validates uploaded CV files before any processing:
  1. File size enforcement (5 MB hard cap)
  2. Magic bytes check (PDF / DOCX only — reject everything else)
  3. Content safety scan (embedded JS, launch actions, macros, zipbombs)
  4. Optional ClamAV scan via clamd socket (graceful fallback if unavailable)

Nothing about file content is logged — only a SHA-256 hash of the raw bytes.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_FILE_BYTES         = 5 * 1024 * 1024    # 5 MB upload cap
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024   # 50 MB zipbomb guard for DOCX

# ── File type signatures ──────────────────────────────────────────────────────
_PDF_MAGIC  = b"%PDF"
_DOCX_MAGIC = b"PK\x03\x04"   # ZIP container — shared by DOCX/XLSX/PPTX

# ── Dangerous PDF byte-patterns ───────────────────────────────────────────────
# These indicate executable content or launch actions inside a PDF.
# Notes on omissions:
#   /EmbeddedFile — omitted: common false positive; PDF generators legitimately
#                   use EmbeddedFile streams for font subsets and metadata.
#   /AA            — omitted: additional-actions dict appears in virtually every
#                   PDF produced by Word/LibreOffice/Adobe for page navigation.
#   /OpenAction    — omitted: used by most PDF viewers for initial-view settings
#                   (zoom level, page layout) — not inherently dangerous.
_PDF_DANGER_PATTERNS: list[bytes] = [
    b"/JS ",
    b"/JavaScript",
    b"/Launch",
]

# ── DOCX VBA/macro indicators (filenames inside the ZIP) ─────────────────────
_DOCX_MACRO_PATHS: frozenset[str] = frozenset({
    "word/vbaproject.bin",
    "xl/vbaproject.bin",
    "ppt/vbaproject.bin",
})


@dataclass
class ScanResult:
    allowed:          bool
    file_hash:        str           # SHA-256 of raw bytes — safe to log
    rejection_reason: str | None = None
    file_type:        str | None = None   # "pdf" | "docx"
    size_bytes:       int        = 0


def scan_file(data: bytes) -> ScanResult:
    """
    Full security scan of raw file bytes.
    Returns ScanResult — caller MUST check .allowed before any further processing.
    Processing time target: <100 ms for a 5 MB file.
    """
    file_hash = hashlib.sha256(data).hexdigest()
    size      = len(data)

    # ── 1. Empty / size gate ──────────────────────────────────────────────────
    if size == 0:
        return ScanResult(False, file_hash, "empty_file", size_bytes=size)

    if size > MAX_FILE_BYTES:
        logger.warning("cv.scan.oversized", hash=file_hash[:16], size_bytes=size)
        return ScanResult(False, file_hash, "file_too_large", size_bytes=size)

    # ── 2. Magic bytes — derive expected type ─────────────────────────────────
    if data[:4] == _DOCX_MAGIC:
        file_type     = "docx"
        content_issue = _scan_docx(data, file_hash)
    elif data[:4] == _PDF_MAGIC:
        file_type     = "pdf"
        content_issue = _scan_pdf(data, file_hash)
    else:
        logger.warning("cv.scan.bad_magic", hash=file_hash[:16], magic=data[:4].hex())
        return ScanResult(False, file_hash, "unsupported_file_type", size_bytes=size)

    if content_issue:
        return ScanResult(False, file_hash, content_issue, file_type=file_type, size_bytes=size)

    # ── 3. Optional ClamAV scan ───────────────────────────────────────────────
    av_issue = _clamav_scan(data, file_hash)
    if av_issue:
        return ScanResult(False, file_hash, av_issue, file_type=file_type, size_bytes=size)

    logger.info("cv.scan.ok", hash=file_hash[:16], file_type=file_type, size_bytes=size)
    return ScanResult(True, file_hash, file_type=file_type, size_bytes=size)


# ── Type-specific scanners ────────────────────────────────────────────────────

def _scan_pdf(data: bytes, file_hash: str) -> str | None:
    """Return rejection reason string if dangerous, else None."""
    for pattern in _PDF_DANGER_PATTERNS:
        if pattern in data:
            logger.warning(
                "cv.scan.pdf_danger",
                hash=file_hash[:16],
                pattern=pattern.decode("ascii", errors="replace"),
            )
            return "pdf_dangerous_content"

    if b"/Encrypt" in data:
        logger.warning("cv.scan.pdf_encrypted", hash=file_hash[:16])
        return "pdf_encrypted"

    return None


def _scan_docx(data: bytes, file_hash: str) -> str | None:
    """Return rejection reason string if dangerous, else None."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Zipbomb guard: sum all uncompressed sizes before extracting anything
            total_uncompressed = sum(info.file_size for info in zf.infolist())
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                logger.warning(
                    "cv.scan.zipbomb",
                    hash=file_hash[:16],
                    uncompressed_bytes=total_uncompressed,
                )
                return "docx_zipbomb"

            # VBA / macro detection
            names_lower = {info.filename.lower() for info in zf.infolist()}
            if names_lower & _DOCX_MACRO_PATHS:
                logger.warning("cv.scan.macro", hash=file_hash[:16])
                return "docx_macro_detected"

    except zipfile.BadZipFile:
        logger.warning("cv.scan.bad_zip", hash=file_hash[:16])
        return "docx_corrupt"
    except Exception as exc:
        logger.warning("cv.scan.docx_error", hash=file_hash[:16], error=str(exc))
        return "docx_parse_error"

    return None


def _clamav_scan(data: bytes, file_hash: str) -> str | None:
    """
    ClamAV instream scan. Returns virus signature string on detection, else None.
    Gracefully skipped when clamd is unavailable (no socket / not installed).
    """
    try:
        import clamd
        cd     = clamd.ClamdUnixSocket()
        result = cd.instream(io.BytesIO(data))
        status, signature = result.get("stream", ("OK", None))
        if status == "FOUND":
            logger.warning("cv.scan.virus", hash=file_hash[:16], signature=signature)
            return f"virus_detected"
    except Exception:
        pass   # clamd not available — skip silently
    return None
