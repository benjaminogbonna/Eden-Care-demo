"""Text matching for the resolver"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Reviewed spelling folds
FOLD = {
    "appendicectomy": "appendectomy", "haemogram": "hemogram", "oesophagitis": "esophagitis", "oesophageal": "esophageal",
    "anaemia": "anemia", "diarrhoea": "diarrhea", "amoxycillin": "amoxicillin", "paediatric": "pediatric",
}
# Words that all mean "a surgical removal"
PROCEDURE_WORDS = {"surgery", "operation", "removed", "removal", "excision", "surgical"}


def normalise(text: str) -> list[str]:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    out = []
    for t in tokens:
        t = FOLD.get(t, t)
        out.append("removal" if t in PROCEDURE_WORDS else t)
    return out


def _contains(seq: list[str], part: list[str]) -> bool:
    n = len(part)
    return any(seq[i:i + n] == part for i in range(len(seq) - n + 1))


def score(query: list[str], term: list[str]) -> float:
    """Similarity of a query to one register term, in [0, 1].

    1.0 exact | 0.92-0.98 the whole term occurs in the query | 0.86 all term tokens present |
    0.5-0.75 the query is only a part of the term (ambiguous by design) | fuzzy: similarity - 0.05
    for near-misses of terms with at least five characters.
    """
    if not query or not term:
        return 0.0
    if query == term:
        return 1.0
    n, m = len(query), len(term)
    if m <= n and _contains(query, term):
        return 0.92 + 0.06 * m / n
    if m >= 2 and set(term) <= set(query):
        return 0.86
    if n < m and (_contains(term, query) or set(query) <= set(term)):
        return 0.5 + 0.25 * n / m
    joined = " ".join(term)
    if len(joined) >= 5:
        best = 0.0
        for width in {m, max(1, m - 1), m + 1}:
            for i in range(0, max(0, n - width) + 1):
                cand = " ".join(query[i:i + width])
                best = max(best, SequenceMatcher(None, cand, joined).ratio())
        if best >= 0.86:
            return round(best - 0.05, 3)
    return 0.0
