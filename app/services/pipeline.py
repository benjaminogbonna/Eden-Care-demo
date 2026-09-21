"""In process pipeline
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.config import Settings, get_settings
from app.core.hashing import sha256_json, sha256_text
from app.core.transcript import parse_transcript
from app.core.errors import ScribeError
from app.services.extraction.service import Engine, extract_note
from app.services.knowledge import build_knowledge, verify_knowledge
from app.services.resolution import resolve_checked
from app.services.resolver import parse_register
from app.services.validation import validate_note


@dataclass
class Loaded:
    """An input that may already have failed to load (missing file, bad encoding)."""
    name: str
    text: str | None = None
    error: str | None = None


@dataclass
class PipelineResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    failed_stage: str | None = None
    failure_message: str | None = None
    exit_code: int = 0

    @property
    def ok(self) -> bool:
        return self.failed_stage is None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


async def run_pipeline(
    transcript: Loaded, register: Loaded, source: Loaded, *,
    engine: Engine | None = None, settings: Settings | None = None, gemini_client: Any | None = None,
    max_chars: int | None = None,
) -> PipelineResult:
    settings = settings or get_settings()
    run_id = uuid.uuid4().hex
    res = PipelineResult()
    entries: dict[str, dict[str, Any]] = {}

    def record(stage: str, status: str, message: str, *, inp: str | None = None, out: str | None = None,
               model: str | None = None, prompt_hash: str | None = None) -> None:
        entries[stage] = {"run_id": run_id, "stage": stage, "timestamp": _now(), "status": status, "input_hash": inp,
                          "output_hash": out, "model": model, "prompt_hash": prompt_hash, "message": message}

    def fail(stage: str, message: str, exit_code: int = 2) -> None:
        if res.failed_stage is None:
            res.failed_stage, res.failure_message, res.exit_code = stage, message, exit_code

    record("transcribe", "skipped", "not implemented in this build: no audio input (text pipeline only)")

    reg = None
    if register.error:
        fail("resolve", f"{register.name}: {register.error}")
    else:
        try:
            reg = parse_register(register.text or "", register.name)
        except ScribeError as exc:
            fail("resolve", exc.message, exc.exit_code)

    note = None
    if res.ok:
        if transcript.error:
            fail("extract", f"{transcript.name}: {transcript.error}")
        else:
            try:
                result = await extract_note(transcript.text or "", engine=engine, settings=settings, gemini_client=gemini_client,
                                            name=transcript.name, max_chars=max_chars)
                note = result.note
                res.outputs["note"] = note
                msg = result.describe() + (f"; {len(result.warnings)} element(s) dropped or flagged" if result.warnings else "")
                record("extract", "ok", msg, inp=sha256_text(transcript.text or ""), out=sha256_json(note),
                       model=(f"google/{result.model}" if result.model else None), prompt_hash=result.prompt_hash)
            except ScribeError as exc:
                fail("extract", f"{transcript.name}: {exc.message}", exc.exit_code)

    if res.ok and note is not None:
        try:
            parsed = parse_transcript(transcript.text or "", name=transcript.name, max_chars=max_chars)
            problems = validate_note(note, parsed)
            if problems:
                fail("validate", f"note rejected: {problems[0]} (+{len(problems) - 1} more)", 1)
            else:
                record("validate", "ok", "note is valid", inp=sha256_json(note), out=sha256_json({"valid": True}))
        except ScribeError as exc:
            fail("validate", exc.message, exc.exit_code)

    if res.ok and note is not None and reg is not None:
        try:
            resolved = resolve_checked(note, reg)
            res.outputs["resolved"] = resolved
            record("resolve", "ok", f"register {register.name}: {len(reg.entries)} entries", inp=sha256_text(sha256_json(note) + (register.text or "")),
                   out=sha256_json(resolved))
        except ScribeError as exc:
            fail("resolve", exc.message, exc.exit_code)

    if res.ok:
        if source.error:
            fail("knowledge", f"{source.name}: {source.error}")
        else:
            try:
                knowledge = build_knowledge(source.text or "", source_name=source.name)
                bad = verify_knowledge(knowledge, source.text or "")
                if bad:
                    fail("knowledge", f"{source.name}: {bad[0]}", 1)
                else:
                    res.outputs["knowledge"] = knowledge
                    record("knowledge", "ok", f"{len(knowledge['rows'])} rows, {len(knowledge['not_in_corpus'])} not_in_corpus",
                           inp=sha256_text(source.text or ""), out=sha256_json(knowledge))
            except ScribeError as exc:
                fail("knowledge", f"{source.name}: {exc.message}", exc.exit_code)

    for stage in ("transcribe", "extract", "validate", "resolve", "knowledge"):
        if stage not in entries:
            if res.failed_stage == stage:
                record(stage, "failed", res.failure_message or "failed")
            else:
                record(stage, "skipped", f"not run: '{res.failed_stage}' failed first" if res.failed_stage else "not run")
    res.log = [entries[s] for s in ("transcribe", "extract", "validate", "resolve", "knowledge")]
    return res
