"""Register loading with strict, located error messages."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import RegisterError

KINDS = ("diagnosis", "drug", "lab", "procedure", "allergen")
REQUIRED_COLUMNS = ("kind", "code", "name", "synonyms")


@dataclass(frozen=True)
class Entry:
    kind: str
    code: str
    name: str
    synonyms: tuple[str, ...]

    @property
    def terms(self) -> tuple[str, ...]:
        return (self.name, *self.synonyms)


@dataclass(frozen=True)
class Register:
    entries: tuple[Entry, ...]
    source: str


def parse_register(text: str, source: str = "register.csv") -> Register:
    if not text.strip():
        raise RegisterError(f"{source}: register file is empty")
    reader = csv.reader(io.StringIO(text), strict=True)
    try:
        header_row = next(reader)
    except csv.Error as exc:
        raise RegisterError(f"{source}, line 1: unparseable header ({exc})") from None
    header = [h.strip().lower() for h in header_row]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise RegisterError(f"{source}: missing required column(s) {missing}; found {header}")
    idx = {c: header.index(c) for c in REQUIRED_COLUMNS}

    entries: list[Entry] = []
    first_seen: dict[str, tuple[str, int]] = {}
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            raise RegisterError(f"{source}, line {reader.line_num}: unparseable row ({exc})") from None
        if not row or all(not c.strip() for c in row):
            continue
        line = reader.line_num
        if len(row) != len(header):
            raise RegisterError(f"{source}, line {line}: expected {len(header)} fields but found {len(row)}")
        kind, code, name = (row[idx[c]].strip() for c in ("kind", "code", "name"))
        if not (kind and code and name):
            raise RegisterError(f"{source}, line {line}: kind, code and name must all be non-empty")
        if kind not in KINDS:
            raise RegisterError(f"{source}, line {line}: unknown kind {kind!r}; expected one of {list(KINDS)}")
        if code in first_seen:
            prev_kind, prev_line = first_seen[code]
            if prev_kind != kind:
                raise RegisterError(
                    f"{source}, line {line}: code {code!r} appears under two different kinds "
                    f"({prev_kind!r} on line {prev_line} and {kind!r})"
                )
            raise RegisterError(f"{source}, line {line}: code {code!r} is duplicated (first on line {prev_line})")
        first_seen[code] = (kind, line)
        synonyms = tuple(s.strip() for s in row[idx["synonyms"]].split(";") if s.strip())
        entries.append(Entry(kind, code, name, synonyms))
    if not entries:
        raise RegisterError(f"{source}: register has a header but no entries")
    return Register(tuple(entries), source)


def load_register(path: str | Path) -> Register:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise RegisterError(f"{p}: register file not found") from None
    except UnicodeDecodeError:
        raise RegisterError(f"{p}: register file is not valid UTF-8") from None
    except OSError as exc:
        raise RegisterError(f"{p}: cannot read register ({exc.strerror})") from None
    return parse_register(text, str(p))
