"""
MarketForge AI — Centralised Configuration.

All settings load from environment variables or a .env file.
Every sub-config is a separate Settings class for clean namespacing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Path constants ────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent.parent.parent  # marketforge-ai/
SRC_DIR     = ROOT_DIR / "src" / "marketforge"
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"
_ENV_FILE   = str(ROOT_DIR / ".env")

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=_ENV_FILE, extra="ignore")

    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY", default="")
    fast_model:     str = "gemini-2.5-flash"
    deep_model:     str = "gemini-2.5-pro"
    cost_cap_usd:   float = 2.00
    temperature:    float = 0.1
    max_retries:    int = 3


class SourceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    adzuna_app_id:  str = ""
    adzuna_app_key: str = ""
    reed_api_key:   str = ""
    tavily_api_key: str = ""

    adzuna_daily_quota: int = 200
    reed_daily_quota:   int = 400
    tavily_daily_quota: int = 100

    request_timeout_s:   float = 25.0
    politeness_delay_s:  float = 3.0    # courtesy delay for scraped boards
    max_jobs_per_source: int   = 100


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    ml_prescreen_enabled:   bool  = True
    ml_prescreen_threshold: float = 0.28
    matchmaker_concurrency: int   = 15
    dedup_hash_ttl_days:    int   = 30    # Redis dedup cache TTL
    llm_cache_ttl_days:     int   = 14    # LLM result cache TTL
    snapshot_cache_ttl_s:   int   = 21600 # 6 hours


class EmailSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMTP_", env_file=_ENV_FILE, extra="ignore")

    host:             str = "smtp.gmail.com"
    port:             int = 587
    user:             str = ""
    password:         str = ""
    recipient_email:  str = Field(default="", validation_alias="REPORT_RECIPIENT_EMAIL")


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    langchain_tracing_v2: str  = Field(default="false", validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key:    str  = Field(default="",      validation_alias="LANGCHAIN_API_KEY")
    langchain_project:    str  = Field(default="marketforge-ai", validation_alias="LANGCHAIN_PROJECT")
    mlflow_tracking_uri:  str  = Field(default="http://localhost:5001", validation_alias="MLFLOW_TRACKING_URI")


class Settings(BaseSettings):
    """Master settings — composes all sub-configs."""
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    llm:     LLMSettings     = Field(default_factory=LLMSettings)
    sources: SourceSettings  = Field(default_factory=SourceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    email:   EmailSettings   = Field(default_factory=EmailSettings)
    obs:     ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    database_url:      str = Field(
        default=f"sqlite+aiosqlite:///{DATA_DIR / 'marketforge.db'}",
        validation_alias="DATABASE_URL",
    )
    database_url_sync: str = Field(
        default=f"sqlite:///{DATA_DIR / 'marketforge.db'}",
        validation_alias="DATABASE_URL_SYNC",
    )
    redis_url:     str  = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    chroma_db_dir: str  = Field(default=str(DATA_DIR / "chroma_db"), validation_alias="CHROMA_DB_DIR")

    environment: Literal["development", "staging", "production"] = "development"
    log_level:   str = "INFO"
    log_format:  Literal["json", "console"] = "console"

    @field_validator("database_url", mode="before")
    @classmethod
    def normalise_async_url(cls, v: str) -> str:
        """Convert Railway's postgresql:// to postgresql+asyncpg:// for async SQLAlchemy."""
        if isinstance(v, str):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            if v.startswith("postgresql://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("database_url_sync", mode="before")
    @classmethod
    def normalise_sync_url(cls, v: str) -> str:
        """Convert async URLs back to sync for non-async operations."""
        if isinstance(v, str):
            v = v.replace("postgres://", "postgresql+psycopg2://", 1)
            v = v.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
            if v.startswith("postgresql://"):
                v = "postgresql+psycopg2://" + v[len("postgresql://"):]
        return v

    @model_validator(mode="after")
    def derive_sync_url_from_async(self) -> "Settings":
        """
        If DATABASE_URL_SYNC was not explicitly provided and database_url is
        PostgreSQL, derive the sync URL automatically so both engines hit the
        same database on Railway (which only auto-sets DATABASE_URL).
        """
        sqlite_default = f"sqlite:///{DATA_DIR / 'marketforge.db'}"
        if (
            self.database_url_sync == sqlite_default
            and "postgresql" in self.database_url
        ):
            sync_url = (
                self.database_url
                .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
                .replace("postgresql+psycopg://", "postgresql+psycopg2://")
            )
            self.database_url_sync = sync_url
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = Settings()
