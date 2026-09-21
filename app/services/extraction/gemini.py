"""Gemini extraction (google-genai).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from app.core.config import PROMPTS_DIR, Settings
from app.core.hashing import sha256_text
from app.core.patterns import is_thinking_aloud
from app.core.text import split_sentences
from app.core.transcript import Transcript
from app.domain.note import CERTAINTIES, KIND_REJECTED, NOT_STATED, SECTIONS
from app.core.errors import ExtractionError
from app.services.validation import _locate, check_element

log = logging.getLogger(__name__)
RETRYABLE = {408, 429, 500, 502, 503, 504}


class LLMElement(BaseModel):
    value: str
    ref: str
    span_text: str
    confidence: float
    certainty: str
    kind: str
    entity: str


LLMNote = create_model("LLMNote", **{s: (list[LLMElement], ...) for s in SECTIONS})  # type: ignore[call-overload]


@dataclass
class GeminiResult:
    note: dict
    warnings: list[str]
    model: str
    prompt_hash: str


def load_prompts() -> tuple[str, str]:
    return (PROMPTS_DIR / "extract_system.md").read_text(encoding="utf-8"), (PROMPTS_DIR / "extract_user.md").read_text(encoding="utf-8")


def prompt_hash() -> str:
    system, user = load_prompts()
    return sha256_text(system + "\n---\n" + user)


def _norm_ref(ref: str) -> str:
    ref = ref.strip()
    return ref if ref.startswith("[") else f"[{ref.strip('[]')}]"


def ground(raw: dict[str, Any], transcript: Transcript) -> tuple[dict, list[str]]:
    """Turn raw model output into a note, dropping (and reporting) everything that is not grounded."""
    warnings: list[str] = []
    note: dict[str, Any] = {}
    for section in SECTIONS:
        kept = []
        for i, e in enumerate(raw.get(section, [])):
            path = f"{section}[{i}]"
            ref, text = _norm_ref(e["ref"]), e["span_text"]
            turn = _locate(transcript, ref, text)
            if turn is None:
                warnings.append(f"dropped {path}: span_text is not verbatim in {ref}")
                continue
            el: dict[str, Any] = {
                "value": e["value"].strip(),
                "span": {"ref": ref, "text": text},
                "confidence": min(1.0, max(0.0, float(e["confidence"]))),
                "attribution": turn.role.lower(),  # never trust the model for who said it
            }
            kind = KIND_REJECTED if e["kind"] == KIND_REJECTED else None
            thinking = any(is_thinking_aloud(s.text) and (text in s.text or s.text in text) for s in split_sentences(turn.text))
            if thinking and kind is None:
                if section in ("assessment", "plan"):
                    kind = KIND_REJECTED
                else:
                    warnings.append(f"dropped {path}: comes from clinician thinking-aloud")
                    continue
            if section == "assessment":
                cert = "differential" if kind else e["certainty"]
                if cert not in CERTAINTIES:
                    warnings.append(f"dropped {path}: invalid certainty {e['certainty']!r}")
                    continue
                el["certainty"] = cert
            if kind:
                el["kind"] = kind
            if e["entity"].strip():
                el["entity"] = e["entity"].strip()
            problems = check_element(section, el, path, transcript)
            if problems:
                warnings.extend(f"dropped {path}: {p.code} - {p.message}" for p in problems)
                continue
            kept.append(el)
        note[section] = kept or NOT_STATED
    return note, warnings


class GeminiExtractor:
    engine = "gemini"

    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        if client is None:
            key = settings.gemini_key_value
            if not key:
                raise ExtractionError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
            try:
                from google import genai
            except ImportError as exc: 
                raise ExtractionError("the google-genai package is not installed") from exc
            client = genai.Client(api_key=key)
        self._client = client

    async def _generate(self, system: str, user: str) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            seed=0,
            response_mime_type="application/json",
            response_schema=LLMNote,
            http_options=types.HttpOptions(timeout=int(self.settings.gemini_timeout_seconds * 1000)),
        )
        attempts = max(1, self.settings.gemini_max_retries)
        for attempt in range(attempts):
            try:
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=self.settings.gemini_model, contents=user, config=config),
                    timeout=self.settings.gemini_timeout_seconds + 5,
                )
            except errors.APIError as exc:
                if exc.code in RETRYABLE and attempt + 1 < attempts:
                    delay = 2**attempt
                    log.warning("Gemini returned %s; retrying in %ss (attempt %s/%s)", exc.code, delay, attempt + 1, attempts)
                    await asyncio.sleep(delay)
                    continue
                raise ExtractionError(f"Gemini API error {exc.code}: {getattr(exc, 'message', str(exc))[:200]}") from exc
            except asyncio.TimeoutError as exc:
                if attempt + 1 < attempts:
                    continue
                raise ExtractionError("Gemini request timed out") from exc
            text = getattr(resp, "text", None)
            if not text:
                raise ExtractionError("Gemini returned an empty response (possibly blocked or truncated)")
            return text
        raise ExtractionError("Gemini request failed after retries")  # pragma: no cover

    async def extract(self, transcript_text: str, transcript: Transcript) -> GeminiResult:
        system, user_tpl = load_prompts()
        user = user_tpl.replace("{transcript}", transcript_text.strip())
        text = await self._generate(system, user)
        try:
            raw = LLMNote.model_validate(json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())).model_dump()
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ExtractionError(f"Gemini output did not match the schema: {str(exc)[:200]}") from exc
        note, warnings = ground(raw, transcript)
        return GeminiResult(note, warnings, self.settings.gemini_model, prompt_hash())
