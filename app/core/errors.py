"""Exception hierarchy. Every error carries a stable code, a CLI exit code and an HTTP status."""

from __future__ import annotations

from typing import Any


class ScribeError(Exception):
    code = "scribe_error"
    exit_code = 1
    http_status = 400

    def __init__(self, message: str, *, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or []


class InputError(ScribeError):
    code = "input_error"
    exit_code = 2
    http_status = 400


class TranscriptError(InputError):
    code = "transcript_error"


class RegisterError(InputError):
    code = "register_error"


class NoteFormatError(InputError):
    code = "note_format_error"


class ExtractionError(ScribeError):
    code = "extraction_error"
    exit_code = 3
    http_status = 502


class NoteRejected(ScribeError):
    """The validator rejected a note. 'details' holds one entry per violation."""

    code = "note_rejected"
    exit_code = 1
    http_status = 422
