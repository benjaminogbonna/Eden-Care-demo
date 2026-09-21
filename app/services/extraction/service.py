"""Engine selection between gemini and rules."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.config import Settings, get_settings
from app.core.transcript import Transcript, parse_transcript
from app.core.errors import ExtractionError
from app.services.extraction.gemini import GeminiExtractor, prompt_hash
from app.services.extraction.rules import RulesExtractor
from app.services.validation import validate_note

log = logging.getLogger(__name__)
Engine = Literal["auto", "gemini", "rules"]


@dataclass
class ExtractionResult:
    note: dict
    engine: str
    model: str | None = None
    prompt_hash: str | None = None
    warnings: list[str] = field(default_factory=list)
    fallback_reason: str | None = None

    def describe(self) -> str:
        base = f"engine={self.engine}" + (f" model={self.model}" if self.model else "")
        return base + (f" (fallback: {self.fallback_reason})" if self.fallback_reason else "")


def _ensure_valid(note: dict, transcript: Transcript, engine: str) -> None:
    problems = validate_note(note, transcript)
    if problems:
        raise ExtractionError(
            f"{engine} engine produced a note that failed validation ({len(problems)} problem(s)): {problems[0]}",
            details=[p.as_dict() for p in problems],
        )


async def _run_rules(transcript: Transcript) -> ExtractionResult:
    note, warnings = RulesExtractor().extract(transcript)
    _ensure_valid(note, transcript, "rules")
    return ExtractionResult(note, "rules", warnings=warnings)


async def _run_gemini(text: str, transcript: Transcript, settings: Settings, client: Any | None) -> ExtractionResult:
    res = await GeminiExtractor(settings, client).extract(text, transcript)
    _ensure_valid(res.note, transcript, "gemini")
    return ExtractionResult(res.note, "gemini", res.model, res.prompt_hash, res.warnings)


async def extract_note(
    transcript_text: str,
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
    gemini_client: Any | None = None,
    name: str = "transcript",
    max_chars: int | None = None,
) -> ExtractionResult:
    settings = settings or get_settings()
    engine = engine or settings.engine
    transcript = parse_transcript(transcript_text, name=name, max_chars=max_chars)

    if engine == "rules":
        return await _run_rules(transcript)
    has_key = bool(settings.gemini_key_value) or gemini_client is not None
    if engine == "gemini":
        if not has_key:
            raise ExtractionError("engine 'gemini' requested but GEMINI_API_KEY is not set")
        return await _run_gemini(transcript_text, transcript, settings, gemini_client)

    # auto
    if not has_key:
        reason = "GEMINI_API_KEY not set"
        log.warning("extract: %s - using the deterministic rules engine (pass --engine rules to silence this)", reason)
    else:
        try:
            return await _run_gemini(transcript_text, transcript, settings, gemini_client)
        except ExtractionError as exc:
            reason = f"Gemini path failed: {exc.message[:160]}"
            log.warning("extract: %s - falling back to the deterministic rules engine", reason)
    result = await _run_rules(transcript)
    result.fallback_reason = reason
    return result
