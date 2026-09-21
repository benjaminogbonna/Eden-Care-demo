"""Text normaliser for speech evaluation. Every reported metric carries NORMALISER_VERSION.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.core.numbers import NUMBER_WORDS, parse_number
from app.core.transcript import Transcript

NORMALISER_VERSION = "scribe-norm/1.0"
LEVELS = ("basic", "numbers", "full")

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+(?:'[^\W\d_]+)*")
_UNIT_FOLD = {
    "milligram": "mg", "milligrams": "mg", "microgram": "mcg", "micrograms": "mcg", "kilogram": "kg", "kilograms": "kg",
    "kilo": "kg", "kilos": "kg", "gram": "g", "grams": "g", "millilitre": "ml", "millilitres": "ml",
    "milliliter": "ml", "milliliters": "ml",
}
FILLERS = frozenset({"um", "uh", "er", "erm", "hmm", "mm", "mhm", "eh", "uhm", "ah"})
VARIANTS = {
    # Swahili
    "nimekua": "nimekuwa", "amekua": "amekuwa", "umekua": "umekuwa", "tumekua": "tumekuwa", "wamekua": "wamekuwa",
    "ahsante": "asante", "dactari": "daktari", "doctari": "daktari", "hujambo": "hujambo",
    # English (British/American, informal)
    "okay": "ok", "okey": "ok", "diarrhoea": "diarrhea", "anaemia": "anemia", "oesophagus": "esophagus",
    "haemoglobin": "hemoglobin", "paediatric": "pediatric", "tumour": "tumor",
}


@dataclass(frozen=True)
class Tok:
    text: str        # normalised token
    raw: str         # the original word(s) it came from
    ref: str         # timestamp of the turn, e.g. "[00:29]"
    role: str
    turn: int


def normalise_words(text: str, level: str = "full") -> list[tuple[str, str]]:
    """Return [(normalised_token, raw_words)] for one string."""
    if level not in LEVELS:
        raise ValueError(f"unknown normaliser level {level!r}")
    s = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    raws = _TOKEN_RE.findall(s)
    orig = _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).replace("\u2019", "'"))
    out: list[tuple[str, str]] = []
    if level == "basic":
        return list(zip(raws, orig))
    i = 0
    while i < len(raws):
        w = raws[i]
        if w in NUMBER_WORDS and w not in ("hundred", "thousand"):
            parsed = parse_number(raws, i)
            if parsed:
                digits, j = parsed
                out.append((digits, " ".join(orig[i:j])))
                i = j
                continue
        out.append((w, orig[i]))
        i += 1
    folded: list[tuple[str, str]] = []
    for k, (t, r) in enumerate(out):
        if t == "over" and 0 < k < len(out) - 1 and out[k - 1][0][0].isdigit() and out[k + 1][0][0].isdigit():
            continue
        if t == "percent" and k > 0 and out[k - 1][0][0].isdigit():
            continue
        folded.append((_UNIT_FOLD.get(t, t), r))
    if level == "numbers":
        return folded
    return [(VARIANTS.get(t, t), r) for t, r in folded if t not in FILLERS]


def normalise_transcript(transcript: Transcript, level: str = "full") -> list[Tok]:
    toks: list[Tok] = []
    for turn in transcript.turns:
        for text, raw in normalise_words(turn.text, level):
            toks.append(Tok(text, raw, turn.ref, turn.role, turn.index))
    return toks
