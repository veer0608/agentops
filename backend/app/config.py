"""Central configuration.

Everything a later phase might want to A/B (provider, model, effort, doc-search
backend, policy thresholds) lives here so an eval run can record the exact config
it used. Pattern mirrors citerag/backend/app/config.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute paths so the DB and .env are found regardless of the process CWD.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DB_PATH = _BACKEND_DIR / "data" / "agentops.db"
_DEFAULT_DATABASE_URL = f"sqlite:///{_DEFAULT_DB_PATH.as_posix()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"), extra="ignore"
    )

    database_url: str = _DEFAULT_DATABASE_URL

    # --- Model provider ----------------------------------------------------
    # The agent core is written against an LLMClient seam, so the provider is a
    # config switch. Without the selected provider's key, the app falls back to
    # the offline DemoLLMClient (deterministic, no network).
    provider: str = "openai"  # "openai" | "anthropic" | "google"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"

    google_api_key: str | None = None
    google_model: str = "gemini-2.0-flash"

    effort: str = "medium"  # Anthropic effort / reserved for reasoning models
    max_agent_steps: int = 8
    runner: str = "manual"  # "manual" | "langgraph"

    # --- Doc search --------------------------------------------------------
    doc_search_mode: str = "stub"  # "stub" | "citerag"
    citerag_url: str = "http://127.0.0.1:8000"

    @property
    def active_model(self) -> str:
        return {
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "google": self.google_model,
        }.get(self.provider, self.openai_model)

    @property
    def has_credentials(self) -> bool:
        return bool(
            {
                "openai": self.openai_api_key,
                "anthropic": self.anthropic_api_key,
                "google": self.google_api_key,
            }.get(self.provider)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
