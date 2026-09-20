"""Request/response schemas."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorItem(BaseModel):
    code: str = Field(examples=["span_not_verbatim"])
    message: str
    field: str | None = Field(default=None, description="JSON path or request field the error refers to")


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T | None = None
    errors: list[ErrorItem] = Field(default_factory=list)


class HealthData(BaseModel):
    service: str
    version: str
    gemini_configured: bool


class ExtractRequest(BaseModel):
    transcript: str = Field(description="Transcript text in the format '[mm:ss] SPEAKER: text', one turn per line")
    engine: Literal["auto", "gemini", "rules"] | None = Field(default=None, description="Defaults to ENGINE")


class ExtractData(BaseModel):
    note: dict[str, Any]
    engine: str
    model: str | None = None
    prompt_hash: str | None = None
    fallback_reason: str | None = Field(default=None, description="Set when Gemini was expected but the rules engine was used")
    warnings: list[str] = Field(default_factory=list)


