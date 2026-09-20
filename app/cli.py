"""Command line interface implementing the contract in Appendix A."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.core.config import DATA_DIR, PROMPTS_DIR, RESOURCES_DIR, get_settings
from app.core.io import dumps, read_json, read_text, write_json, write_text
from app.core.transcript import parse_transcript
from app.core.errors import InputError, ScribeError
from app.core.logging_config import configure_logging
from app.services.extraction.service import extract_note
from app.services.knowledge import build_knowledge, verify_knowledge
from app.services.pipeline import Loaded, run_pipeline
from app.services.resolution import resolve_checked
from app.services.resolver import parse_register
from app.services.speech_eval.metrics import evaluate_texts
from app.services.validation import validate_note

VERSION = "1.0.0"


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


async def cmd_check(_: argparse.Namespace) -> int:
    problems: list[str] = []
    notes: list[str] = []
    if sys.version_info < (3, 10):
        problems.append(f"Python >= 3.10 required (found {sys.version.split()[0]})")
    for mod, required in (("fastapi", True), ("uvicorn", True), ("pydantic", True), ("pydantic_settings", True), ("google.genai", False)):
        try:
            __import__(mod)
        except ImportError:
            (problems if required else notes).append(f"python package {mod!r} is not installed (pip install -r requirements.txt)")
    for path in (PROMPTS_DIR / "extract_system.md", PROMPTS_DIR / "extract_user.md", RESOURCES_DIR / "swahili_words.txt",
                 RESOURCES_DIR / "clinical_terms.txt", RESOURCES_DIR / "english_guard_words.txt"):
        if not path.exists():
            problems.append(f"missing bundled file: {path}")
    if not problems:
        try:
            sample = "[00:00] DOCTOR: Any allergies?\n[00:03] PATIENT: Penicillin.\n"
            result = await extract_note(sample, engine="rules")
            if validate_note(result.note, parse_transcript(sample)):
                problems.append("self-test: rules engine produced a note that fails validation")
        except Exception as exc:  
            problems.append(f"self-test failed: {exc}")
    settings = get_settings()
    if settings.gemini_key_value:
        notes.append(f"GEMINI_API_KEY is set: `extract` will use Gemini ({settings.gemini_model}) unless --engine rules/--offline is given")
    else:
        notes.append("GEMINI_API_KEY is not set: `extract` will fall back to the deterministic rules engine and say so (use --offline to make that explicit)")
    for n in notes:
        print(f"note: {n}")
    if problems:
        for p in problems:
            _err(f"MISSING: {p}")
        return 1
    print(f"scribe {VERSION}: environment ready")
    return 0


def _engine(args: argparse.Namespace) -> str | None:
    return "rules" if getattr(args, "offline", False) else getattr(args, "engine", None)


async def cmd_extract(args: argparse.Namespace) -> int:
    text = await read_text(args.transcript)
    result = await extract_note(text, engine=_engine(args), name=args.transcript)
    await write_json(args.out, result.note)
    n = sum(len(v) for v in result.note.values() if isinstance(v, list))
    print(f"extract: wrote {args.out} ({n} elements; {result.describe()})", file=sys.stderr)
    for w in result.warnings:
        _err(f"extract: warning: {w}")
    return 0


async def cmd_validate(args: argparse.Namespace) -> int:
    transcript = parse_transcript(await read_text(args.transcript), name=args.transcript)
    note = await read_json(args.note)
    problems = validate_note(note, transcript)
    if problems:
        _err(f"REJECTED: {len(problems)} violation(s) in {args.note}")
        for p in problems:
            _err(f"  {p}")
        return 1
    print(f"OK: {args.note} is valid against {args.transcript}")
    return 0


async def cmd_resolve(args: argparse.Namespace) -> int:
    note = await read_json(args.note)
    register = parse_register(await read_text(args.register), args.register)
    await write_json(args.out, resolve_checked(note, register))
    print(f"resolve: wrote {args.out}", file=sys.stderr)
    return 0


async def cmd_knowledge(args: argparse.Namespace) -> int:
    source = await read_text(args.source)
    knowledge = build_knowledge(source, source_name=args.source)
    problems = verify_knowledge(knowledge, source)
    if problems:
        raise ScribeError(f"knowledge self-check failed: {problems[0]}")
    await write_json(args.out, knowledge)
    print(f"knowledge: wrote {args.out} ({len(knowledge['rows'])} rows)", file=sys.stderr)
    return 0


async def cmd_speech_eval(args: argparse.Namespace) -> int:
    metrics = await evaluate_texts(await read_text(args.ref), await read_text(args.hyp), ref_name=args.ref, hyp_name=args.hyp)
    await write_json(args.out, metrics)
    print(f"speech-eval: WER {metrics['wer_overall']:.4f}  CER {metrics['cer_overall']:.4f}  clinical-token error rate "
          f"{metrics['clinical_token_error_rate']:.4f}  role accuracy {metrics['role_accuracy']:.4f}  (normaliser {metrics['normaliser_version']})")
    return 0


async def _load(path: str) -> Loaded:
    try:
        return Loaded(path, await read_text(path))
    except InputError as exc:
        return Loaded(path, None, exc.message)


async def cmd_pipeline(args: argparse.Namespace) -> int:
    transcript, register, source = await asyncio.gather(_load(args.transcript), _load(args.register), _load(args.source))
    result = await run_pipeline(transcript, register, source, engine=_engine(args))
    out = Path(args.out)
    for key, fname in (("note", "note.json"), ("resolved", "resolved.json"), ("knowledge", "knowledge.json")):
        if key in result.outputs:
            await write_json(out / fname, result.outputs[key])
    await write_text(out / "run_log.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in result.log))
    if not result.ok:
        _err(f"pipeline FAILED at stage '{result.failed_stage}': {result.failure_message}")
        return result.exit_code or 1
    print(f"pipeline: ok -> {out}/ (note.json, resolved.json, knowledge.json, run_log.jsonl)", file=sys.stderr)
    return 0


async def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    config = uvicorn.Config("app.api.main:app", host=args.host or settings.host, port=args.port or settings.port,
                            log_level=settings.log_level.lower())
    await uvicorn.Server(config).serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scribe", description="Grounded clinical scribe and speech-evaluation tooling")
    sub = p.add_subparsers(dest="command", required=True)

    def engine_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--engine", choices=["auto", "gemini", "rules"], help="extraction engine (default: ENGINE, i.e. auto)")
        sp.add_argument("--offline", action="store_true", help="no network: force the deterministic rules engine")

    sub.add_parser("check", help="verify the environment").set_defaults(func=cmd_check)
    sp = sub.add_parser("extract", help="transcript -> note.json")
    sp.add_argument("--transcript", required=True); sp.add_argument("--out", required=True); engine_flags(sp)
    sp.set_defaults(func=cmd_extract)
    sp = sub.add_parser("validate", help="prove note.json is grounded in the transcript (exit 0 = valid)")
    sp.add_argument("--transcript", required=True); sp.add_argument("--note", required=True)
    sp.set_defaults(func=cmd_validate)
    sp = sub.add_parser("resolve", help="note.json + register.csv -> resolved.json (no model, no network)")
    sp.add_argument("--note", required=True); sp.add_argument("--register", required=True); sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_resolve)
    sp = sub.add_parser("knowledge", help="guideline excerpt -> cited knowledge rows")
    sp.add_argument("--source", required=True); sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_knowledge)
    sp = sub.add_parser("speech-eval", help="reference + hypothesis transcript -> metrics.json")
    sp.add_argument("--ref", required=True); sp.add_argument("--hyp", required=True); sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_speech_eval)
    sp = sub.add_parser("pipeline", help="run every stage; writes run_log.jsonl; non-zero on any failure")
    sp.add_argument("--transcript", required=True); sp.add_argument("--register", required=True)
    sp.add_argument("--source", required=True); sp.add_argument("--out", required=True); engine_flags(sp)
    sp.set_defaults(func=cmd_pipeline)
    sp = sub.add_parser("serve", help="run the REST API")
    sp.add_argument("--host"); sp.add_argument("--port", type=int)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)
    try:
        return asyncio.run(args.func(args))
    except ScribeError as exc:
        _err(f"error: {exc.message}")
        return exc.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _err(f"internal error: {type(exc).__name__}: {exc}")
        return 70
