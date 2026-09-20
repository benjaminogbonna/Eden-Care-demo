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


class ValidateRequest(BaseModel):
    transcript: str
    note: dict[str, Any]


class ValidateData(BaseModel):
    valid: bool


class ResolveRequest(BaseModel):
    note: dict[str, Any]
    register_csv: str | None = Field(default=None, description="Register CSV text; defaults to the bundled register")


class ResolveData(BaseModel):
    resolved: dict[str, Any]


class KnowledgeRequest(BaseModel):
    source: str | None = Field(default=None, description="Guideline text with a 'SOURCE: ..., Section n.n, page n.' header; defaults to the bundled excerpt")


class SpeechEvalRequest(BaseModel):
    reference: str
    hypothesis: str


class PipelineRequest(BaseModel):
    transcript: str
    register_csv: str | None = None
    source: str | None = None
    engine: Literal["auto", "gemini", "rules"] | None = None


class PipelineData(BaseModel):
    ok: bool
    note: dict[str, Any] | None = None
    resolved: dict[str, Any] | None = None
    knowledge: dict[str, Any] | None = None
    run_log: list[dict[str, Any]]
