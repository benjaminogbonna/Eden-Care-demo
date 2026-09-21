"""Rule/wordlist language tagger for code-switched Kenyan clinical speech: en | sw | other.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import RESOURCES_DIR

METHOD = (
    "wordlist + morphology rules: digits, single letters and drug/lab names -> other; bundled Swahili wordlist or "
    "Swahili verb morphology (subject prefix + tense marker + stem) -> sw; everything else -> en"
)
_MORPH = re.compile(r"^(?:ni|u|a|tu|m|wa|si|hu|ha|hatu|ham|hawa)(?:me|li|na|ta|ki|ku|ka|sha|nge|ja)[a-z]{2,}[aeiu]$")


def _load(name: str) -> frozenset[str]:
    return frozenset(w.strip() for w in (RESOURCES_DIR / name).read_text(encoding="utf-8").splitlines() if w.strip())


@lru_cache(maxsize=1)
def _lists() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    return _load("swahili_words.txt"), _load("english_guard_words.txt"), _load("clinical_terms.txt")


def clinical_terms() -> frozenset[str]:
    return _lists()[2]


def tag(token: str) -> str:
    sw, guard, clinical = _lists()
    if token[0].isdigit() or len(token) == 1 or token in clinical:
        return "other"
    if token in sw:
        return "sw"
    if len(token) >= 6 and token not in guard and _MORPH.match(token):
        return "sw"
    return "en"
