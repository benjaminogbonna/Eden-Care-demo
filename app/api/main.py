"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import settings_dep
from app.api.routes import VERSION, health_router, router
from app.api.schemas import ApiResponse, ErrorItem
from app.core.config import Settings, get_settings
from app.core.errors import ScribeError
from app.core.logging_config import configure_logging

log = logging.getLogger("scribe.api")


def _envelope(status: int, message: str, errors: list[ErrorItem]) -> JSONResponse:
    body = ApiResponse[None](success=False, message=message, data=None, errors=errors)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_logging(settings.log_level)
        log.info("scribe API starting (gemini configured: %s, auth: %s)", bool(settings.gemini_key_value), settings.api_key is not None)
        yield

    app = FastAPI(
        title="Eden Care API",
        version=VERSION,
        summary="Clinical extraction, coding and speech evaluation",
        description=(
            "Every response has the shape {success, message, data, errors}. Notes are validated against the transcript"
        ),
        lifespan=lifespan,
    )
    app.dependency_overrides[settings_dep] = lambda: settings  # the factory's settings are authoritative
    if settings.cors_origin_list:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
            return _envelope(413, "request body too large", [ErrorItem(code="body_too_large", message=f"limit is {settings.max_body_bytes} bytes")])
        return await call_next(request)

    @app.exception_handler(ScribeError)
    async def scribe_error(_: Request, exc: ScribeError):
        errors = [ErrorItem(code=d.get("code", exc.code), message=d.get("message", ""), field=d.get("path")) for d in exc.details] \
            or [ErrorItem(code=exc.code, message=exc.message)]
        return _envelope(exc.http_status, exc.message, errors)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError):
        errors = [ErrorItem(code="invalid_request", message=e["msg"], field=".".join(str(p) for p in e["loc"])) for e in exc.errors()]
        return _envelope(422, "request validation failed", errors)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return _envelope(exc.status_code, str(exc.detail), [ErrorItem(code=f"http_{exc.status_code}", message=str(exc.detail))])

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unhandled error: %s", type(exc).__name__)
        return _envelope(500, "internal server error", [ErrorItem(code="internal_error", message="an unexpected error occurred")])

    app.include_router(health_router)
    app.include_router(router)
    return app


app = create_app()
