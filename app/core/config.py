# configuration

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoice, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).parent.parent.parent
PROMPT_DIR = REPO_ROOT / 'prompts'
RESOURCES_DIR = Path(__file__).resolve().parent / 'resources'

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
        populate_by_name=True
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoice('GEMINI_API_KEY', 'GOOGLE_API_KEY')
    )
    gemini_model: str = 'gemini-3.5-flash'
    gemini_time_out_seconds: float = 90.0
    gemini_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()