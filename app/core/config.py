"""Environment configuration."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False, populate_by_name=True
    )

    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    gemini_model: str = "gemini-3.5-flash"
    gemini_timeout_seconds: float = 90.0
    gemini_max_retries: int = 3

    engine: Literal["auto", "gemini", "rules"] = "auto"
    log_level: str = "INFO"

    api_key: SecretStr | None = None
    cors_origins: str = ""
    max_body_bytes: int = 2_000_000
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def gemini_key_value(self) -> str | None:
        if self.gemini_api_key is None:
            return None
        value = self.gemini_api_key.get_secret_value().strip()
        return value or None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
