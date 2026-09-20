"""Parser for the consultation transcript format:  [mm:ss] SPEAKER: text"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.errors import TranscriptError

TURN_RE = re.compile(
    r"^\s*\[(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})\]\s*(?P<role>[A-Za-z_]+)\s*:\s*(?P<text>.*?)\s*$"
)
CLINICIAN_ROLES = frozenset({"DOCTOR", "NURSE"})
HISTORY_ROLES = frozenset({"PATIENT", "COMPANION"})


@dataclass(frozen=True)
class Turn:
    index: int
    ref: str 
    seconds: int
    role: str
    text: str


@dataclass
class Transcript:
    turns: list[Turn]
    _by_ref: dict[str, list[Turn]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for t in self.turns:
            self._by_ref.setdefault(t.ref, []).append(t)

    def lines_for(self, ref: str) -> list[Turn]:
        return self._by_ref.get(ref, [])

    def __len__(self) -> int:
        return len(self.turns)


def _to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def parse_transcript(text: str, *, name: str = "transcript", max_chars: int | None = None) -> Transcript:
    """Parse text into turns.
    """
    if max_chars is not None and len(text) > max_chars:
        raise TranscriptError(f"{name} is too large ({len(text)} characters; limit {max_chars})")
    turns: list[list] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        m = TURN_RE.match(raw)
        if m:
            ts = m.group("ts")
            turns.append([f"[{ts}]", _to_seconds(ts), m.group("role").upper(), m.group("text")])
        elif turns:
            turns[-1][3] = (turns[-1][3] + " " + raw.strip()).strip()
        else:
            raise TranscriptError(
                f"{name}, line {lineno}: expected '[mm:ss] SPEAKER: text' but found {raw.strip()[:60]!r}"
            )
    if not turns:
        raise TranscriptError(f"{name} contains no '[mm:ss] SPEAKER: text' turns")
    return Transcript([Turn(i, r, s, role, t) for i, (r, s, role, t) in enumerate(turns)])
