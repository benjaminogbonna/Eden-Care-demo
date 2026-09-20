"""Spoken number handling shared by the validator"""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SWAHILI_NUMBERS = {"moja": 1, "mbili": 2, "tatu": 3, "nne": 4, "tano": 5, "sita": 6, "saba": 7, "nane": 8, "tisa": 9, "kumi": 10}
NUMBER_WORDS = frozenset(_UNITS) | frozenset(_TENS) | {"hundred", "thousand"}


UNIT_WORDS = frozenset(
    "mg milligram milligrams mcg microgram micrograms g gram grams kg kilogram kilograms ml millilitre millilitres "
    "milliliter milliliters litre litres tablet tablets tab tabs capsule capsules puff puffs drop drops unit units "
    "day days week weeks month months year years hour hours minute minutes time times dose doses percent".split()
)


def _below_100(tokens: list[str], i: int) -> tuple[int, int] | None:
    if i >= len(tokens):
        return None
    t = tokens[i]
    if t in _TENS:
        v, j = _TENS[t], i + 1
        if j < len(tokens) and tokens[j] in _UNITS and 1 <= _UNITS[tokens[j]] <= 9:
            v, j = v + _UNITS[tokens[j]], j + 1
        return v, j
    if t in _UNITS:
        return _UNITS[t], i + 1
    return None


def _below_1000(tokens: list[str], i: int) -> tuple[int, int] | None:
    first = _below_100(tokens, i)
    if first is None:
        return None
    v, j = first
    if j < len(tokens) and tokens[j] == "hundred" and v < 10:
        v, j = v * 100, j + 1
        k = j + 1 if j < len(tokens) and tokens[j] == "and" else j
        rest = _below_100(tokens, k)
        if rest is not None:
            v, j = v + rest[0], rest[1]
    return v, j


def parse_number(tokens: list[str], i: int) -> tuple[str, int] | None:
    """Parse a spoken English number starting at tokens[i] (lower-case words).

    Returns (digit_string, next_index) or None. Handles 'twenty nineteen' -> 2019,
    'nineteen ninety five' -> 1995, 'two thousand and nineteen' -> 2019, 'one hundred and
    twenty eight' -> 128 and 'thirty six point eight' -> 36.8.
    """
    first = _below_100(tokens, i)
    if first is None:
        return None
    v, j = first
    # year pairs: only when the opening pair is 19xx / 20xx. eg, so that "twenty one" stays 21
    if v in (19, 20) and tokens[i] in ("nineteen", "twenty"):
        if j < len(tokens) and tokens[j] in ("oh", "zero"):
            unit = _below_100(tokens, j + 1)
            if unit and 1 <= unit[0] <= 9:
                return str(v * 100 + unit[0]), unit[1]
        second = _below_100(tokens, j)
        if second and second[0] >= 10:
            return str(v * 100 + second[0]), second[1]
    full = _below_1000(tokens, i)
    assert full is not None
    v, j = full
    if j < len(tokens) and tokens[j] == "thousand":
        v, j = v * 1000, j + 1
        k = j + 1 if j < len(tokens) and tokens[j] == "and" else j
        rest = _below_1000(tokens, k)
        if rest is not None:
            v, j = v + rest[0], rest[1]
    out = str(v)
    if j < len(tokens) and tokens[j] == "point":
        k, digits = j + 1, ""
        while k < len(tokens) and tokens[k] in _UNITS and _UNITS[tokens[k]] <= 9:
            digits += str(_UNITS[tokens[k]])
            k += 1
        if digits:
            out, j = f"{out}.{digits}", k
    return out, j


_WORD_RE = re.compile(r"[A-Za-z']+|\d+(?:\.\d+)?")
_DIGITS_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def digit_sequences(text: str) -> set[str]:
    return {m.group().replace(",", "") for m in _DIGITS_RE.finditer(text)}


def spoken_number_runs(text: str) -> list[tuple[int, int, str]]:
    """(start, end, digits) for every run of English number words in `text`."""
    matches = list(_WORD_RE.finditer(text))
    words = [m.group().lower() for m in matches]
    runs: list[tuple[int, int, str]] = []
    i = 0
    while i < len(words):
        if words[i] in NUMBER_WORDS and words[i] not in ("hundred", "thousand"):
            parsed = parse_number(words, i)
            if parsed:
                digits, j = parsed
                # only join words separated by spaces/hyphens
                ok = all(re.fullmatch(r"[\s-]*", text[matches[k].end() : matches[k + 1].start()]) for k in range(i, j - 1))
                if ok:
                    runs.append((matches[i].start(), matches[j - 1].end(), digits))
                    i = j
                    continue
        i += 1
    return runs


def numeric_values(text: str) -> set[str]:
    """Every number in `text`, whether written as digits, English words or Swahili numerals."""
    values = digit_sequences(text)
    values.update(d for _, _, d in spoken_number_runs(text))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in SWAHILI_NUMBERS:
            values.add(str(SWAHILI_NUMBERS[w]))
    return values


def digitize(text: str) -> str:
    """Write spoken numbers as digits when they quantify a unit ('four weeks' -> '4 weeks').

    Also folds 'milligrams' to 'mg' and 'N over M' to 'N/M'. Numbers not followed by a unit word
    are left as spoken, so 'one of them' is never rewritten.
    """
    out, last = [], 0
    for start, end, digits in spoken_number_runs(text):
        nxt = re.match(r"\s+([A-Za-z]+)", text[end:])
        if nxt and nxt.group(1).lower() in UNIT_WORDS:
            out.append(text[last:start])
            out.append(digits)
            last = end
    out.append(text[last:])
    s = "".join(out)
    s = re.sub(r"\bmilligrams?\b", "mg", s, flags=re.I)
    s = re.sub(r"\bmicrograms?\b", "mcg", s, flags=re.I)
    s = re.sub(r"\b(\d{2,3})\s+over\s+(\d{2,3})\b", r"\1/\2", s, flags=re.I)
    return s


def spoken_to_digits(text: str) -> str:
    """Unconditionally rewrite runs of spoken numbers as digits (used to read vitals)."""
    out, last = [], 0
    for start, end, digits in spoken_number_runs(text):
        out.append(text[last:start])
        out.append(digits)
        last = end
    out.append(text[last:])
    return "".join(out)
