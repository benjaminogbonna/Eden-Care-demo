"""Regexes shared by extraction, grounding and validation"""

from __future__ import annotations

import re

ICD_RE = re.compile(r"[A-Z]\d{2}(\.\d)?")
ATC_RE = re.compile(r"[A-Z]\d{2}[A-Z]{2}\d{2}")
ECODE_RE = re.compile(r"(LAB|PGP|ALG)-EC-\d{4}")
CODE_PATTERNS = (("ICD", ICD_RE), ("ATC", ATC_RE), ("EDEN_CARE", ECODE_RE))


def find_code_like(text: str) -> list[tuple[str, str]]:
    hits = []
    for name, rx in CODE_PATTERNS:
        for m in rx.finditer(text):
            hits.append((name, m.group()))
    return hits


THINKING_RES = [
    re.compile(p, re.I)
    for p in (
        r"\bi was (?:just )?(?:wonder(?:ing)?|thinking|considering|about to)\b",
        r"\bi (?:wonder|am wondering|am thinking)\b",
        r"\blet'?s not go (?:there|down that)\b",
        r"\bnot going (?:there|down that)\b",
        r"\bthinking (?:aloud|out loud)\b",
        r"\bscratch that\b",
        r"\bnever mind\b",
        r"\bon second thought\b",
        r"\bfor a (?:moment|second|minute),? i (?:thought|wondered)\b",
        r"\bbut no\b",
        r"\bno,? wait\b",
        r"\blet me think\b",
        r"\bhmm+\b",
        r"\bactually,? no\b",
    )
]


def is_thinking_aloud(sentence: str) -> bool:
    return any(rx.search(sentence) for rx in THINKING_RES)


_REL = r"(?:father|mother|brother|sister|grandfather|grandmother|grandparents?|parents?|uncle|aunt|cousin|son|daughter)"
_SW_REL = r"(?:baba|mama|kaka|dada|babu|bibi|mjomba|shangazi)"
FAMILY_WORD_RE = re.compile(rf"\b{_REL}s?\b|\b{_SW_REL}\s+(?:yangu|wangu|wake|wetu)\b", re.I)
FAMILY_RE = re.compile(FAMILY_WORD_RE.pattern + r"|\bfamily history\b|\bin the family\b", re.I)
# Relation words accepted while extracting family history
FAMILY_TOPIC_REL_RE = re.compile(rf"\b(?:{_REL}s?|{_SW_REL})\b", re.I)

NEGATION_ONLY_RE = re.compile(
    r"^\s*(?:no|nope|hapana|la|hakuna|sina|nothing|none|not really)\b[\s,.!]*(?:none|nothing|hapana|sina|at all)?[\s.!]*$", re.I
)
ACK_RE = re.compile(
    r"^\s*(?:(?:yes|yeah|ndiyo|ndio|okay|ok|sawa|poa|asante|thanks|thank you|alright|right|mm+|hmm+|eh|aha|pole)"
    r"(?:[\s,.!]+(?:doctor|daktari|sana|asante|sawa|mama|baba|nurse|sister))*)[\s.!]*$",
    re.I,
)
