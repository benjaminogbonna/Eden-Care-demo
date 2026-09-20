"""Async-friendly file helpers."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.core.errors import InputError


def _read_text_sync(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise InputError(f"file not found: {path}") from None
    except IsADirectoryError:
        raise InputError(f"expected a file but found a directory: {path}") from None
    except UnicodeDecodeError as exc:
        raise InputError(f"file is not valid UTF-8 text: {path} ({exc.reason})") from None
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror}") from None


async def read_text(path: str | Path) -> str:
    return await asyncio.to_thread(_read_text_sync, Path(path))


async def read_json(path: str | Path) -> Any:
    text = await read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from None


def _write_text_sync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def write_text(path: str | Path, text: str) -> None:
    await asyncio.to_thread(_write_text_sync, Path(path), text)


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


async def write_json(path: str | Path, obj: Any) -> None:
    await write_text(path, dumps(obj))
