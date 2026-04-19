"""
MarketForge AI — FastAPI Security Middleware

Integrates Department 8 (Security & Guardrails) with the FastAPI app.
All user-facing endpoints pass through this middleware before any LLM call.

Usage:
    from api.security import SecurityMiddleware, require_clean_input
    app.add_middleware(SecurityMiddleware)
"""
from __future__ import annotations

import time
from typing import Callable

import structlog
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# ── Rate limiting state (backed by Redis when available) ─────────────────────

_rate_limit_fallback: dict[str, list[float]] = {}   # in-memory fallback


def _check_rate_limit(ip: str, limit: int, window_seconds: int) -> bool:
    """
    Returns True if the request is within the rate limit, False if exceeded.
    Uses Redis RateLimiter if available, falls back to in-memory dict.
    """
    try:
        from marketforge.memory.redis_cache import RateLimiter
        return RateLimiter().is_allowed(f"api:{ip}", limit=limit, window_seconds=window_seconds)
    except Exception:
        # In-memory fallback for local dev / Redis unavailable
        now = time.time()
        cutoff = now - window_seconds
        history = _rate_limit_fallback.get(ip, [])
        history = [t for t in history if t > cutoff]
        if len(history) >= limit:
            return False
        history.append(now)
        _rate_limit_fallback[ip] = history
        return True


# Rate limiting is handled by rate_limit_middleware in main.py (per-endpoint keys).
# SecurityMiddleware focuses on headers, logging, and exception handling only.
_RATE_LIMITS: dict[str, tuple[int, int]] = {}

# Endpoints that run the security validation pipeline
_SECURITY_VALIDATED_PATHS = {"/api/v1/career/analyse"}


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that enforces:
    1. Per-IP rate limiting (Redis-backed, in-memory fallback)
    2. Request logging with threat context
    3. Security headers on all responses

    Note: The deep input validation (injection detection, PII scrubbing)
    is handled by SecurityLeadAgent inside each endpoint handler, not here.
    This middleware handles the HTTP layer only — fast, synchronous checks.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        client_ip = self._get_client_ip(request)
        path = request.url.path

        # ── Rate limiting ─────────────────────────────────────────────────────
        limit_config = _RATE_LIMITS.get(path)
        if limit_config:
            limit, window = limit_config
            if not _check_rate_limit(client_ip, limit, window):
                logger.warning(
                    "security.rate_limited",
                    ip=client_ip,
                    path=path,
                    limit=limit,
                    window_s=window,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Too many requests. Limit: {limit} per {window}s.",
                        "retry_after_seconds": window,
                    },
                    headers={"Retry-After": str(window)},
                )

        # ── Process request ───────────────────────────────────────────────────
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error("security.unhandled_exception", path=path, error=str(exc))
            return JSONResponse(
                status_code=500,
                content={"error": "internal_server_error", "message": "An unexpected error occurred."},
            )

        # ── Security headers ──────────────────────────────────────────────────
        duration_ms = round((time.time() - start) * 1000)
        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]           = "DENY"
        response.headers["X-XSS-Protection"]          = "1; mode=block"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"]   = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
        response.headers["X-Response-Time-Ms"]        = str(duration_ms)
        response.headers["Cache-Control"]             = "no-store" if path in _SECURITY_VALIDATED_PATHS else "public, max-age=300"

        if duration_ms > 10_000:
            logger.warning("security.slow_request", path=path, ip=client_ip, ms=duration_ms)

        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract real client IP. Railway appends the real client IP as the rightmost XFF entry."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            return parts[-1] if parts else "unknown"
        return request.client.host if request.client else "unknown"


# ── Dependency injection helpers for endpoint handlers ───────────────────────

async def validate_career_input(
    raw_input: dict,
    request: Request,
) -> tuple[bool, dict, str | None]:
    """
    Dependency used by the career advisor endpoint.
    Runs the full SecurityLeadAgent pipeline on user input.
    Returns (accepted, sanitised_input, rejection_reason).
    """
    client_ip = SecurityMiddleware._get_client_ip(request)
    try:
        from marketforge.agents.security.lead_agent import validate_user_input
        return await validate_user_input(raw_input, source_ip=client_ip)
    except Exception as exc:
        logger.error("security.validate_career_input.failed", error=str(exc))
        # Fail closed — reject on error rather than passing unsafe input
        return False, {}, "security_check_failed"


def require_clean_input(accepted: bool, rejection_reason: str | None) -> None:
    """
    Raises HTTPException 400 if the security validation rejected the input.
    Call this after validate_career_input() in the endpoint handler.
    """
    if not accepted:
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "input_rejected",
                "reason":  rejection_reason or "security_validation_failed",
                "message": "Your input could not be processed. Please check your input and try again.",
            },
        )


async def sanitise_output(text: str) -> tuple[str, list[str]]:
    """
    Run output guardrails (PII scrub, salary sanity check) before serving.
    Returns (sanitised_text, warnings).
    """
    try:
        from marketforge.agents.security.guardrails import validate_output
        return validate_output(text)
    except Exception as exc:
        logger.warning("security.sanitise_output.failed", error=str(exc))
        return text, []
