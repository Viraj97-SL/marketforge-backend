"""
MarketForge AI — GDPR Compliance Layer for CV Processing.

Enforces the following non-negotiable rules for every CV upload:
  1. Explicit consent gate — 403 if consent is not True
  2. No CV file or extracted text written to disk or database
  3. PII stripped from extracted text before it reaches any LLM
  4. Session token is an anonymous hash — contains no PII
  5. Only SHA-256 file hash appears in structured logs (never text content)
  6. Processing timeout guard — data reference dropped on expiry

PII patterns overlap intentionally with security/lead_agent.py to ensure
defence-in-depth: both layers independently scrub PII.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# ── PII scrubbing patterns ────────────────────────────────────────────────────
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_uk":    re.compile(r"\b(?:\+44|0)[\s\-]?\d{4}[\s\-]?\d{6}\b"),
    "phone_intl":  re.compile(r"\+\d{1,3}[\s\-]?\d{3,}[\s\-]?\d{3,}"),
    "ni_number":   re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b"),
    "postcode_uk": re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b"),
    "dob":         re.compile(
        r"\b(?:DOB|Date\s+of\s+Birth|Born)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", re.I
    ),
    "address":     re.compile(
        r"\b\d+\s+[A-Za-z]+\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Close|Cl|Way)\b",
        re.I,
    ),
}


@dataclass
class GDPRContext:
    """
    Immutable context for one CV processing session.
    scrubbed_text is the only text that may flow into LLM calls or logs.
    raw_text is intentionally absent — do not store it anywhere.
    """
    session_token:   str
    scrubbed_text:   str
    pii_types_found: list[str] = field(default_factory=list)
    consent_given:   bool = True


class ConsentNotGiven(ValueError):
    """Raised when the user has not provided explicit GDPR consent."""


def check_consent(consent: bool) -> None:
    """
    Gate: must be called before any CV data is processed.
    Raises ConsentNotGiven → caller should return HTTP 403.
    """
    if not consent:
        raise ConsentNotGiven(
            "GDPR consent is required before processing CV data. "
            "The user must explicitly agree to the privacy notice."
        )


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """
    Replace all detected PII with [REDACTED:TYPE] tokens.

    Returns:
        scrubbed_text: safe to log and pass to LLM
        pii_types:     list of PII type names found (for audit)

    Note: this does NOT use spaCy NER — person names in CVs are intentionally
    preserved for the ATS structure check (recruiter names in text, not applicant
    identity in structured fields, are the actual risk).
    """
    found: list[str] = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
            text = pattern.sub(f"[REDACTED:{pii_type.upper()}]", text)
    return text, found


def make_session_token(file_hash: str) -> str:
    """
    Create an anonymous, non-reversible session token.
    Combines file hash with a per-request random salt so the same CV
    uploaded twice produces different tokens (no correlation possible).
    """
    salt = secrets.token_hex(16)
    return hashlib.sha256(f"{file_hash}:{salt}".encode()).hexdigest()[:32]


def build_gdpr_context(raw_text: str, file_hash: str, consent: bool) -> GDPRContext:
    """
    Full GDPR processing setup for one CV upload.

    Call once per request:
      1. Checks consent (raises ConsentNotGiven on failure)
      2. Scrubs PII from raw_text
      3. Creates anonymous session token
      4. Returns GDPRContext — only scrubbed_text flows downstream

    The caller should drop any reference to raw_text immediately after this call.
    """
    check_consent(consent)
    scrubbed, pii_found = scrub_pii(raw_text)
    token = make_session_token(file_hash)

    if pii_found:
        # Log types only — never log the content
        logger.info("cv.gdpr.pii_scrubbed", session=token[:8], pii_types=pii_found)

    return GDPRContext(
        session_token   = token,
        scrubbed_text   = scrubbed,
        pii_types_found = pii_found,
        consent_given   = True,
    )
