"""
MarketForge AI — Redis Cache Layer.

Provides LLM result caching, dedup set, rate-limit counters, and
dashboard cache. All operations degrade gracefully if Redis is unavailable
(falls back to the PostgreSQL llm_cache table).
"""
from __future__ import annotations

import json
import time
from typing import Any

import structlog

from marketforge.config.settings import settings

logger = structlog.get_logger(__name__)


def _get_client():
    """Lazily create a Redis client; return None if Redis is unreachable."""
    try:
        import redis
        client = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=2)
        client.ping()
        return client
    except Exception as exc:
        logger.warning("redis.unavailable", error=str(exc))
        return None


_redis_client = None


def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = _get_client()
    else:
        try:
            _redis_client.ping()
        except Exception:
            _redis_client = _get_client()
    return _redis_client


# ── LLM Result Cache ──────────────────────────────────────────────────────────
class LLMCache:
    """
    Caches LLM outputs keyed on (input hash).
    Primary: Redis with TTL.
    Fallback: market.llm_cache PostgreSQL table.
    """

    TTL_SECONDS = settings.pipeline.llm_cache_ttl_days * 86_400

    def get(self, cache_key: str) -> dict | None:
        r = get_redis()
        if r:
            try:
                raw = r.get(f"market:llm:{cache_key}")
                if raw:
                    logger.debug("llm_cache.hit.redis", key=cache_key[:12])
                    return json.loads(raw)
            except Exception:
                pass
        return self._pg_get(cache_key)

    def set(self, cache_key: str, value: dict) -> None:
        r = get_redis()
        if r:
            try:
                r.setex(f"market:llm:{cache_key}", self.TTL_SECONDS, json.dumps(value))
                return
            except Exception:
                pass
        self._pg_set(cache_key, value)

    # ── PostgreSQL fallback ───────────────────────────────────────────────────
    def _pg_get(self, cache_key: str) -> dict | None:
        try:
            from datetime import datetime
            from sqlalchemy import text
            from marketforge.memory.postgres import get_sync_engine
            engine = get_sync_engine()
            is_sqlite = engine.dialect.name == "sqlite"
            table = "llm_cache" if is_sqlite else "market.llm_cache"
            with engine.connect() as conn:
                row = conn.execute(
                    text(f"SELECT result_json FROM {table} WHERE cache_key=:k AND expires_at > :now"),
                    {"k": cache_key, "now": datetime.utcnow().isoformat()},
                ).fetchone()
            if row:
                logger.debug("llm_cache.hit.postgres", key=cache_key[:12])
                return json.loads(row[0])
        except Exception as exc:
            logger.warning("llm_cache.pg_get.error", error=str(exc))
        return None

    def _pg_set(self, cache_key: str, value: dict) -> None:
        try:
            from datetime import datetime, timedelta
            from sqlalchemy import text
            from marketforge.memory.postgres import get_sync_engine
            engine = get_sync_engine()
            is_sqlite = engine.dialect.name == "sqlite"
            table = "llm_cache" if is_sqlite else "market.llm_cache"
            expires = (datetime.utcnow() + timedelta(seconds=self.TTL_SECONDS)).isoformat()
            with engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {table} (cache_key, result_json, expires_at)
                    VALUES (:k, :v, :exp)
                    ON CONFLICT(cache_key) DO UPDATE
                    SET result_json=EXCLUDED.result_json, expires_at=EXCLUDED.expires_at
                """), {"k": cache_key, "v": json.dumps(value), "exp": expires})
                conn.commit()
        except Exception as exc:
            logger.warning("llm_cache.pg_set.error", error=str(exc))


# ── Dedup Cache ───────────────────────────────────────────────────────────────
class RedisDedup:
    """
    Fast deduplication using a Redis set per week.
    Falls back to the DedupStore PostgreSQL table.
    """

    TTL_SECONDS = settings.pipeline.dedup_hash_ttl_days * 86_400

    def _key(self) -> str:
        from datetime import date
        week = date.today().isocalendar()
        return f"market:dedup:{week.year}:{week.week}"

    def is_seen(self, dedup_hash: str) -> bool:
        r = get_redis()
        if r:
            try:
                return bool(r.sismember(self._key(), dedup_hash))
            except Exception:
                pass
        return False

    def mark_seen(self, dedup_hash: str) -> None:
        r = get_redis()
        if r:
            try:
                key = self._key()
                r.sadd(key, dedup_hash)
                r.expire(key, self.TTL_SECONDS)
            except Exception:
                pass

    def bulk_filter_new(self, hashes: list[str]) -> list[str]:
        """Return hashes not yet in the dedup set; mark all as seen."""
        r = get_redis()
        if not r:
            return hashes   # can't filter without Redis — let DB handle it
        try:
            pipe = r.pipeline()
            key = self._key()
            for h in hashes:
                pipe.sismember(key, h)
            results = pipe.execute()
            new = [h for h, seen in zip(hashes, results) if not seen]
            if new:
                pipe2 = r.pipeline()
                pipe2.sadd(key, *new)
                pipe2.expire(key, self.TTL_SECONDS)
                pipe2.execute()
            return new
        except Exception:
            return hashes


# ── Rate Limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Per-IP / per-source rate limiter using Redis sliding window.
    Falls back to a simple in-process dict if Redis is unavailable
    (not distributed, but prevents abuse in single-process deployments).
    """

    _fallback: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        r = get_redis()
        if r:
            try:
                now = time.time()
                window_key = f"market:ratelimit:{key}"
                pipe = r.pipeline()
                pipe.zremrangebyscore(window_key, 0, now - window_seconds)
                pipe.zcard(window_key)
                pipe.zadd(window_key, {str(now): now})
                pipe.expire(window_key, window_seconds)
                _, count, *_ = pipe.execute()
                return int(count) < limit
            except Exception:
                pass
        # Fallback: in-process sliding window
        now = time.time()
        timestamps = [t for t in self._fallback.get(key, []) if now - t < window_seconds]
        if len(timestamps) >= limit:
            return False
        timestamps.append(now)
        self._fallback[key] = timestamps
        return True


# ── Dashboard Cache ───────────────────────────────────────────────────────────
class DashboardCache:
    """Simple read-through cache for dashboard queries."""

    TTL = settings.pipeline.snapshot_cache_ttl_s

    def get(self, key: str) -> Any | None:
        r = get_redis()
        if not r:
            return None
        try:
            raw = r.get(f"market:dashboard:{key}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        r = get_redis()
        if not r:
            return
        try:
            r.setex(f"market:dashboard:{key}", self.TTL, json.dumps(value, default=str))
        except Exception:
            pass

    def invalidate(self, prefix: str = "") -> None:
        r = get_redis()
        if not r:
            return
        try:
            pattern = f"market:dashboard:{prefix}*"
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=pattern, count=100)
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass
