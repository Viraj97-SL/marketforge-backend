"""
MarketForge AI — Bootstrap Script

Run this once before the first pipeline execution:
  python scripts/bootstrap.py

Tasks:
  1. Initialise all database tables (idempotent)
  2. Seed the skill taxonomy into market.skill_taxonomy
  3. Verify connectivity (PostgreSQL, Redis, LLM API)
  4. Print a setup summary
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketforge.utils.logger import setup_logging
setup_logging()

import structlog
log = structlog.get_logger("bootstrap")


def init_db() -> bool:
    try:
        from marketforge.memory.postgres import init_database
        init_database()
        log.info("bootstrap.db.ok")
        return True
    except Exception as exc:
        log.error("bootstrap.db.failed", error=str(exc))
        return False


def seed_taxonomy() -> int:
    try:
        from marketforge.nlp.taxonomy import SKILL_TAXONOMY
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        table     = "skill_taxonomy" if is_sqlite else "market.skill_taxonomy"

        import json
        seeded = 0
        with engine.connect() as conn:
            for entry in SKILL_TAXONOMY:
                try:
                    if is_sqlite:
                        conn.execute(text(f"""
                            INSERT OR IGNORE INTO {table} (canonical, aliases, category)
                            VALUES (:canonical, :aliases, :category)
                        """), {
                            "canonical": entry["canonical"],
                            "aliases":   json.dumps(entry.get("aliases", [])),
                            "category":  entry.get("category", "general"),
                        })
                    else:
                        conn.execute(text(f"""
                            INSERT INTO {table} (canonical, aliases, category)
                            VALUES (:canonical, :aliases::text[], :category)
                            ON CONFLICT(canonical) DO NOTHING
                        """), {
                            "canonical": entry["canonical"],
                            "aliases":   entry.get("aliases", []),
                            "category":  entry.get("category", "general"),
                        })
                    seeded += 1
                except Exception:
                    pass
            conn.commit()

        log.info("bootstrap.taxonomy.seeded", count=seeded)
        return seeded
    except Exception as exc:
        log.error("bootstrap.taxonomy.failed", error=str(exc))
        return 0


def check_redis() -> bool:
    try:
        from marketforge.memory.redis_cache import get_redis
        r = get_redis()
        if r:
            r.ping()
            log.info("bootstrap.redis.ok")
            return True
        log.warning("bootstrap.redis.unavailable")
        return False
    except Exception as exc:
        log.warning("bootstrap.redis.failed", error=str(exc))
        return False


def check_llm() -> bool:
    try:
        from marketforge.config.settings import settings
        if not settings.llm.gemini_api_key:
            log.warning("bootstrap.llm.no_key")
            return False
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        llm = ChatGoogleGenerativeAI(
            model=settings.llm.fast_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=0,
        )
        resp = llm.invoke([HumanMessage(content="Reply with the single word: ready")])
        ok   = "ready" in resp.content.lower()
        if ok:
            log.info("bootstrap.llm.ok", model=settings.llm.fast_model)
        else:
            log.warning("bootstrap.llm.unexpected_response", response=resp.content[:50])
        return ok
    except Exception as exc:
        log.error("bootstrap.llm.failed", error=str(exc))
        return False


def check_api_keys() -> dict[str, bool]:
    from marketforge.config.settings import settings
    return {
        "gemini":  bool(settings.llm.gemini_api_key),
        "adzuna":  bool(settings.sources.adzuna_app_id and settings.sources.adzuna_app_key),
        "reed":    bool(settings.sources.reed_api_key),
        "tavily":  bool(settings.sources.tavily_api_key),
        "smtp":    bool(settings.email.user and settings.email.password),
    }


def main() -> None:
    print("\n" + "═" * 55)
    print("  MarketForge AI — Bootstrap")
    print("═" * 55)

    results = {
        "database":  init_db(),
        "taxonomy":  seed_taxonomy() > 0,
        "redis":     check_redis(),
        "llm_api":   check_llm(),
    }
    api_keys = check_api_keys()

    print("\n📦 System checks:")
    for check, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {check}")

    print("\n🔑 API keys configured:")
    for key, ok in api_keys.items():
        print(f"  {'✅' if ok else '⚠️ '}  {key}")

    all_ok = all(results.values())
    print("\n" + "═" * 55)
    if all_ok:
        print("  ✅  Bootstrap complete — ready to run pipeline")
        print("\nNext steps:")
        print("  1. Start services:     docker-compose up -d")
        print("  2. Start FastAPI:      uvicorn api.main:app --reload")
        print("  3. Start dashboard:    streamlit run dashboard/app.py")
        print("  4. Trigger first run:  python scripts/run_pipeline.py")
    else:
        print("  ⚠️   Bootstrap completed with warnings — check logs above")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
