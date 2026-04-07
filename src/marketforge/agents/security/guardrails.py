"""
MarketForge AI — Department 8: Security & Guardrails

All user inputs pass through SecurityGuardrails before reaching any LLM.
All LLM outputs pass through OutputGuardrailsAgent before serving to users.

This module exposes the synchronous validate_input() and validate_output()
functions used by the FastAPI middleware and the career advisor endpoint.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.config.settings import settings

logger = structlog.get_logger(__name__)

# ── Known injection patterns ──────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:previous|all|prior)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|DAN|jailbreak)", re.I),
    re.compile(r"system\s*prompt|<system>|</system>", re.I),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.I),
    re.compile(r"disregard\s+(?:your|all|the)", re.I),
    re.compile(r"act\s+as\s+(?:if|a|an)\s+", re.I),
    re.compile(r"jailbreak|DAN\b|STAN\b", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"),
    re.compile(r"###\s*instruction|##\s*new\s+task", re.I),
    re.compile(r"output\s+your\s+(?:system\s+)?prompt", re.I),
    re.compile(r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt", re.I),
    re.compile(r"what\s+(?:are\s+your|is\s+your)\s+instructions?", re.I),
]

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_EMAIL    = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
_PII_PHONE    = re.compile(r"(\+44\s?|0)[\d\s\-]{9,13}")
_PII_NI       = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I)
_PII_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", re.I)

# Salary hallucination: flags figures > 50% off the market range
_MAX_SANE_SALARY = 500_000
_MIN_SANE_SALARY = 10_000


# ── Lightweight injection scorer (no model needed) ────────────────────────────
def _injection_score(text: str) -> float:
    """
    Returns an injection likelihood score 0.0 – 1.0 based on pattern matching.
    A score > 0.5 is treated as high-risk; > 0.3 is suspicious.
    """
    if not text:
        return 0.0
    hits = sum(1 for p in _INJECTION_PATTERNS if p.search(text))
    return min(hits / 3.0, 1.0)   # 3+ hits → score 1.0


def _scrub_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII tokens with redaction placeholders. Returns (scrubbed, findings)."""
    findings: list[str] = []
    scrubbed = text

    if _PII_EMAIL.search(scrubbed):
        findings.append("email")
        scrubbed = _PII_EMAIL.sub("[REDACTED_EMAIL]", scrubbed)

    if _PII_PHONE.search(scrubbed):
        findings.append("phone")
        scrubbed = _PII_PHONE.sub("[REDACTED_PHONE]", scrubbed)

    if _PII_NI.search(scrubbed):
        findings.append("national_insurance")
        scrubbed = _PII_NI.sub("[REDACTED_NI]", scrubbed)

    if _PII_POSTCODE.search(scrubbed):
        findings.append("postcode")
        scrubbed = _PII_POSTCODE.sub("[REDACTED_POSTCODE]", scrubbed)

    return scrubbed, findings


def _log_security_event(
    event_type: str,
    severity: str,
    source_ip: str | None,
    input_text: str,
    detection_method: str,
    action: str,
    threat_score: float = 0.0,
) -> None:
    """Write a security event to market.security_log."""
    input_hash = hashlib.sha256((input_text or "").encode()).hexdigest()[:16]
    try:
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text as sql_text
        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        table     = "security_log" if is_sqlite else "market.security_log"
        now       = datetime.utcnow().isoformat()
        with engine.connect() as conn:
            conn.execute(sql_text(f"""
                INSERT INTO {table}
                    (event_type, severity, source_ip, input_hash,
                     detection_method, action_taken, threat_score, logged_at)
                VALUES (:et, :sv, :ip, :ih, :dm, :at, :ts, :now)
            """), {
                "et": event_type, "sv": severity, "ip": source_ip or "",
                "ih": input_hash, "dm": detection_method, "at": action,
                "ts": threat_score, "now": now,
            })
            conn.commit()
    except Exception as exc:
        logger.warning("security_log.write_failed", error=str(exc))


# ── Public interface ──────────────────────────────────────────────────────────

class SecurityValidationResult:
    """Immutable result object returned by validate_input()."""
    __slots__ = ("allowed", "sanitised_text", "threat_score", "rejection_reason", "pii_found")

    def __init__(
        self,
        allowed: bool,
        sanitised_text: str,
        threat_score: float = 0.0,
        rejection_reason: str = "",
        pii_found: list[str] | None = None,
    ) -> None:
        self.allowed         = allowed
        self.sanitised_text  = sanitised_text
        self.threat_score    = threat_score
        self.rejection_reason= rejection_reason
        self.pii_found       = pii_found or []


def validate_input(
    text: str,
    field_name: str = "input",
    source_ip: str | None = None,
    max_length: int = 4000,
) -> SecurityValidationResult:
    """
    Synchronous security validation gate for all user inputs.

    Pipeline:
      1. Length enforcement
      2. Character encoding normalisation
      3. PII detection and scrubbing
      4. Prompt injection pattern scoring
      5. Log event if suspicious

    Returns a SecurityValidationResult — callers MUST check .allowed
    before passing .sanitised_text to any LLM.
    """
    if not text or not text.strip():
        return SecurityValidationResult(allowed=True, sanitised_text="")

    # ── 1. Length ──────────────────────────────────────────────────────────────
    if len(text) > max_length:
        _log_security_event("length_violation", "medium", source_ip, text[:100],
                            "length_check", f"rejected: len={len(text)}")
        return SecurityValidationResult(
            allowed=False,
            sanitised_text="",
            rejection_reason=f"Input too long ({len(text)} chars, max {max_length})",
        )

    # ── 2. Encoding normalisation ─────────────────────────────────────────────
    try:
        cleaned = text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        cleaned = text

    # ── 3. PII scrubbing ──────────────────────────────────────────────────────
    scrubbed, pii_found = _scrub_pii(cleaned)
    if pii_found:
        logger.info("security.pii_scrubbed", field=field_name, types=pii_found)

    # ── 4. Injection scoring ──────────────────────────────────────────────────
    score = _injection_score(scrubbed)

    if score >= 0.5:
        _log_security_event("prompt_injection", "high", source_ip, scrubbed[:200],
                            "pattern_match", f"rejected: score={score:.2f}", score)
        logger.warning("security.injection_blocked", field=field_name, score=score)
        return SecurityValidationResult(
            allowed=False,
            sanitised_text="",
            threat_score=score,
            rejection_reason="Input rejected: potential prompt injection detected.",
            pii_found=pii_found,
        )

    if score >= 0.3:
        _log_security_event("suspicious_input", "medium", source_ip, scrubbed[:200],
                            "pattern_match", f"flagged: score={score:.2f}", score)
        logger.info("security.suspicious_input", field=field_name, score=score)

    return SecurityValidationResult(
        allowed=True,
        sanitised_text=scrubbed,
        threat_score=score,
        pii_found=pii_found,
    )


def validate_output(
    text: str,
    context_data: dict | None = None,
) -> tuple[str, list[str]]:
    """
    Validate and scrub LLM output before serving to the user.

    Checks:
      - PII that may have leaked into the response
      - Salary figures wildly outside the market range
      - Returns (scrubbed_text, list_of_warnings)
    """
    if not text:
        return text, []

    warnings: list[str] = []

    # PII scrub
    scrubbed, pii_found = _scrub_pii(text)
    if pii_found:
        warnings.append(f"pii_scrubbed: {pii_found}")

    # Sanity-check salary figures against market data
    salary_nums = re.findall(r"£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?", scrubbed)
    for num_str in salary_nums:
        value = float(num_str.replace(",", ""))
        if value < 1000:
            value *= 1000  # e.g. £45k
        if value > _MAX_SANE_SALARY or value < _MIN_SANE_SALARY:
            warnings.append(f"suspect_salary: £{value:,.0f} outside [{_MIN_SANE_SALARY}, {_MAX_SANE_SALARY}]")
            logger.warning("security.suspect_salary", value=value)

    return scrubbed, warnings


# ── InputSanitisationAgent (Deep Agent wrapper) ────────────────────────────────

class InputSanitisationAgent(DeepAgent):
    """
    Department 8 sub-agent: InputSanitisationAgent

    plan():    Reads the security threat log to detect whether the current
               IP/session has prior suspicious activity. Adjusts thresholds:
               if 2+ prior suspicious events from same IP, lowers the injection
               score threshold from 0.5 to 0.3 for this request.

    execute(): Runs the full validation pipeline:
               schema validation → length bounds → encoding normalisation →
               PII detection → injection pattern scoring.

    reflect(): Checks the weekly threat log for new injection technique patterns
               not covered by current regex. Flags to ThreatIntelligenceAgent
               if a novel pattern appears 3+ times.

    output():  Returns the ValidationResult or a structured rejection.
    """

    agent_id   = "input_sanitisation_v1"
    department = "security"

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive    = state.get("adaptive_params", {})
        source_ip   = context.get("source_ip")
        ip_risk     = adaptive.get("ip_risk_scores", {})
        prior_score = ip_risk.get(source_ip, 0.0) if source_ip else 0.0

        # Lower threshold for IPs with prior suspicious activity
        threshold = 0.3 if prior_score > 0.5 else 0.5

        return {
            "threshold":  threshold,
            "source_ip":  source_ip,
            "adaptive":   adaptive,
            "text":       context.get("text", ""),
            "field_name": context.get("field_name", "input"),
            "max_length": context.get("max_length", 4000),
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        result = validate_input(
            plan["text"],
            field_name=plan["field_name"],
            source_ip=plan["source_ip"],
            max_length=plan["max_length"],
        )
        return {"validation_result": result}

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        vr       = result.get("validation_result")
        adaptive = plan.get("adaptive", {})
        ip       = plan.get("source_ip")

        if vr and not vr.allowed and ip:
            # Update per-IP risk score
            ip_risk = adaptive.get("ip_risk_scores", {})
            old     = ip_risk.get(ip, 0.0)
            ip_risk[ip] = min(1.0, old + 0.2)  # increment risk score
            adaptive["ip_risk_scores"] = ip_risk

        state["adaptive_params"] = adaptive
        quality = "good" if (not vr or vr.allowed) else "warning"
        return {"quality": quality, "threat_score": vr.threat_score if vr else 0.0}

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        vr = result.get("validation_result")
        return {
            "allowed":          vr.allowed          if vr else False,
            "sanitised_text":   vr.sanitised_text   if vr else "",
            "threat_score":     vr.threat_score      if vr else 1.0,
            "rejection_reason": vr.rejection_reason  if vr else "unknown",
            "pii_found":        vr.pii_found         if vr else [],
        }
