"""API routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import require_api_key, settings_dep
from app.api.schemas import (
    ApiResponse, ErrorItem, ExtractData, ExtractRequest, HealthData, KnowledgeRequest, PipelineData, PipelineRequest,
    ResolveData, ResolveRequest, SpeechEvalRequest, ValidateData, ValidateRequest,
)
from app.core.config import DATA_DIR, Settings
from app.core.io import read_text
from app.core.transcript import parse_transcript
from app.core.errors import NoteRejected
from app.services.extraction.service import extract_note
from app.services.knowledge import build_knowledge, verify_knowledge
from app.services.pipeline import Loaded, run_pipeline
from app.services.resolution import resolve_checked
from app.services.resolver import parse_register
from app.services.speech_eval.metrics import evaluate_texts
from app.services.validation import validate_note

VERSION = "1.0.0"
health_router = APIRouter(tags=["health"])
router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
SettingsDep = Annotated[Settings, Depends(settings_dep)]

ERROR_RESPONSES = {
    400: {"model": ApiResponse[None], "description": "Malformed input"},
    401: {"model": ApiResponse[None], "description": "Missing or invalid X-API-Key"},
    422: {"model": ApiResponse[None], "description": "Validation failed"},
    502: {"model": ApiResponse[None], "description": "Upstream (Gemini) failure"},
}


@health_router.get("/health", response_model=ApiResponse[HealthData], summary="Liveness and configuration check")
async def health(settings: SettingsDep) -> ApiResponse[HealthData]:
    return ApiResponse(success=True, message="ok", data=HealthData(service="scribe", version=VERSION, gemini_configured=bool(settings.gemini_key_value)))


@router.post("/extract", response_model=ApiResponse[ExtractData], responses=ERROR_RESPONSES, summary="Transcript -> grounded note")
async def extract(body: ExtractRequest, settings: SettingsDep) -> ApiResponse[ExtractData]:
    result = await extract_note(body.transcript, engine=body.engine, settings=settings, max_chars=settings.max_body_bytes)
    return ApiResponse(success=True, message=f"note extracted ({result.describe()})", data=ExtractData(
        note=result.note, engine=result.engine, model=result.model, prompt_hash=result.prompt_hash,
        fallback_reason=result.fallback_reason, warnings=result.warnings))


@router.post("/validate", response_model=ApiResponse[ValidateData], responses=ERROR_RESPONSES, summary="Prove a note is grounded in its transcript")
async def validate(body: ValidateRequest, settings: SettingsDep) -> ApiResponse[ValidateData]:
    transcript = parse_transcript(body.transcript, max_chars=settings.max_body_bytes)
    problems = validate_note(body.note, transcript)
    if problems:
        raise NoteRejected(f"note rejected with {len(problems)} violation(s)", details=[p.as_dict() for p in problems])
    return ApiResponse(success=True, message="note is valid", data=ValidateData(valid=True))


@router.post("/resolve", response_model=ApiResponse[ResolveData], responses=ERROR_RESPONSES, summary="Deterministic coding of a note (no model, no network)")
async def resolve(body: ResolveRequest) -> ApiResponse[ResolveData]:
    text = body.register_csv if body.register_csv is not None else await read_text(DATA_DIR / "register.csv")
    register = parse_register(text, "request.register_csv" if body.register_csv is not None else "data/register.csv")
    return ApiResponse(success=True, message="note resolved", data=ResolveData(resolved=resolve_checked(body.note, register)))


@router.post("/knowledge", response_model=ApiResponse[dict], responses=ERROR_RESPONSES, summary="Cited knowledge rows from a guideline excerpt")
async def knowledge(body: KnowledgeRequest) -> ApiResponse[dict]:
    source = body.source if body.source is not None else await read_text(DATA_DIR / "guideline.txt")
    result = build_knowledge(source, source_name="request.source")
    problems = verify_knowledge(result, source)
    if problems:
        raise NoteRejected("knowledge rows failed verification", details=[{"code": "quote_not_in_source", "message": p} for p in problems])
    return ApiResponse(success=True, message=f"{len(result['rows'])} rows", data=result)


@router.post("/speech-eval", response_model=ApiResponse[dict], responses=ERROR_RESPONSES, summary="WER/CER, clinical-token error rate, role accuracy, error table")
async def speech_eval(body: SpeechEvalRequest) -> ApiResponse[dict]:
    metrics = await evaluate_texts(body.reference, body.hypothesis)
    return ApiResponse(success=True, message="evaluation complete", data=metrics)


@router.post("/pipeline", response_model=ApiResponse[PipelineData], responses=ERROR_RESPONSES, summary="Run extract -> validate -> resolve -> knowledge")
async def pipeline(body: PipelineRequest, settings: SettingsDep):
    register_text = body.register_csv if body.register_csv is not None else await read_text(DATA_DIR / "register.csv")
    source_text = body.source if body.source is not None else await read_text(DATA_DIR / "guideline.txt")
    result = await run_pipeline(Loaded("request.transcript", body.transcript), Loaded("request.register_csv", register_text),
                                Loaded("request.source", source_text), engine=body.engine, settings=settings,
                                max_chars=settings.max_body_bytes)
    data = PipelineData(ok=result.ok, note=result.outputs.get("note"), resolved=result.outputs.get("resolved"),
                        knowledge=result.outputs.get("knowledge"), run_log=result.log)
    if result.ok:
        return ApiResponse(success=True, message="pipeline completed", data=data)
    payload = ApiResponse[PipelineData](success=False, message=f"pipeline failed at stage '{result.failed_stage}'", data=data,
                                        errors=[ErrorItem(code="stage_failed", message=result.failure_message or "", field=result.failed_stage)])
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))
