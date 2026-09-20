"""Checked wrapper around the resolver: rejects malformed notes before resolving."""

from __future__ import annotations

from typing import Any

from app.domain.note import SECTIONS
from app.core.errors import NoteFormatError
from app.services.resolver import Register, resolve_note


def resolve_checked(note: Any, register: Register) -> dict[str, Any]:
    if not isinstance(note, dict):
        raise NoteFormatError("note must be a JSON object with the 13 section keys")
    unknown = [k for k in note if k not in SECTIONS]
    if unknown:
        raise NoteFormatError(f"note has unknown section(s): {unknown}")
    for section, body in note.items():
        if isinstance(body, list):
            for i, el in enumerate(body):
                if not isinstance(el, dict) or not isinstance(el.get("value"), str):
                    raise NoteFormatError(f"note.{section}[{i}] must be an object with a string 'value'")
        elif body != "NOT_STATED":
            raise NoteFormatError(f"note.{section} must be NOT_STATED or a list of elements")
    return resolve_note(note, register)
