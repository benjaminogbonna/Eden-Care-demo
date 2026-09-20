"""Offset preserving text splitting. Fragments are always exact substrings of the source, which is
what lets every extracted element cite verbatim words."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Frag:
    text: str
    start: int
    end: int


_ABBREV = {"dr", "mr", "mrs", "ms", "vs", "approx", "st", "e.g", "i.e"}


def _trim(text: str, start: int, end: int) -> Frag | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Frag(text[start:end], start, end) if end > start else None


def split_sentences(text: str) -> list[Frag]:
    """Split on . ! ? followed by whitespace/end, but not after initials ("H. pylori") or abbreviations."""
    frags: list[Frag] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?=\s|$)", text):
        before = text[start : m.start()]
        tail = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
        last = tail.group(1).lower() if tail else ""
        nxt = text[m.end() :].lstrip()[:1]
        if m.group() == "." and (last in _ABBREV or (len(last) == 1 and nxt.islower())):
            continue
        f = _trim(text, start, m.end())
        if f:
            frags.append(f)
        start = m.end()
    f = _trim(text, start, len(text))
    if f:
        frags.append(f)
    return frags


def split_regex(text: str, pattern: str, *, base: int = 0) -> list[Frag]:
    """Split `text` on a delimiter regex, returning trimmed fragments with absolute offsets."""
    frags: list[Frag] = []
    pos = 0
    for m in re.finditer(pattern, text, flags=re.I):
        f = _trim(text, pos, m.start())
        if f:
            frags.append(Frag(f.text, f.start + base, f.end + base))
        pos = m.end()
    f = _trim(text, pos, len(text))
    if f:
        frags.append(Frag(f.text, f.start + base, f.end + base))
    return frags
