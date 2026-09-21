"""Transcript validator.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.core.numbers import numeric_values
from app.core.patterns import FAMILY_RE, FAMILY_WORD_RE, find_code_like, is_thinking_aloud
from app.core.text import split_sentences
from app.core.transcript import CLINICIAN_ROLES, Transcript, Turn
from app.domain.note import ATTRIBUTIONS, CERTAINTIES, KIND_REJECTED, NOT_STATED, SECTIONS

CLINICIAN_ONLY_SECTIONS = frozenset({"assessment", "plan", "examination"})
_NEG_START = re.compile(r"^\s*(?:no|not|denies|without|nil|negative|absent)\b", re.I)
_ROS_STOP = frozenset(
    "no not denies without nil negative absent patient reports reported is are was ni normal fine sawa the of in a and any "
    "has have had been says said".split()
)


@dataclass
class Violation:
    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def __str__(self) -> str:
        return f"[{self.code}] {self.path}: {self.message}"


def _walk_strings(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.<key {k!r}>", str(k)
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _locate(transcript: Transcript, ref: str, text: str) -> Turn | None:
    for turn in transcript.lines_for(ref):
        if text in turn.text:
            return turn
    return None


def _prefixes(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in _ROS_STOP and len(w) > 2}


def _asked_by_doctor(transcript: Transcript, turn: Turn, value: str) -> bool:
    """A ROS negative is only valid if the nearest preceding clinician turn was a question about it."""
    for prev in reversed(transcript.turns[: turn.index]):
        if prev.role in CLINICIAN_ROLES:
            return "?" in prev.text and bool(_prefixes(value) & _prefixes(prev.text))
    return False


def check_element(section: str, el: Any, path: str, transcript: Transcript, *, scan_codes: bool = True) -> list[Violation]:
    """All per-element rules. Also used to gate model output before it becomes a note.

    `scan_codes` is False when the caller (validate_note) has already scanned every string in the note.
    """
    v: list[Violation] = []
    if not isinstance(el, dict):
        return [Violation("structure", path, "element must be an object")]
    if scan_codes:
        for label, text_ in _walk_strings(el, path):
            v.extend(Violation("code_like_string", label, f"{name}-style code {code!r} is not allowed in a note") for name, code in find_code_like(text_))
    value, span, conf = el.get("value"), el.get("span"), el.get("confidence")
    if not isinstance(value, str) or not value.strip():
        v.append(Violation("structure", path, "value must be a non-empty string"))
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        v.append(Violation("confidence_range", path, "confidence must be a number between 0 and 1"))
    if not isinstance(span, dict) or not isinstance(span.get("ref"), str) or not isinstance(span.get("text"), str) or not span["text"].strip():
        v.append(Violation("structure", path, "span must be {ref, text} with non-empty text"))
        return v
    if not isinstance(value, str):
        return v
    ref, text = span["ref"], span["text"]
    if not transcript.lines_for(ref):
        return v + [Violation("span_ref_unknown", path, f"span.ref {ref} is not a line in the transcript")]
    turn = _locate(transcript, ref, text)
    if turn is None:
        return v + [Violation("span_not_verbatim", path, f"span.text is not a verbatim substring of {ref}: {text[:60]!r}")]

    # Numbers: every number in value (or entity) must appear in the span, in digits or spoken form.
    entity = el.get("entity") if isinstance(el.get("entity"), str) else ""
    missing = sorted((numeric_values(value) | numeric_values(entity)) - numeric_values(text))
    if missing:
        v.append(Violation("number_not_in_span", path, f"number(s) {missing} in value are not in the cited span {text!r}"))

    kind, attribution = el.get("kind"), el.get("attribution")
    if kind is not None and kind != KIND_REJECTED:
        v.append(Violation("structure", path, f"kind must be {KIND_REJECTED!r} when present"))
    if attribution is not None:
        if not isinstance(attribution, str) or attribution not in ATTRIBUTIONS:
            v.append(Violation("structure", path, f"attribution must be one of {ATTRIBUTIONS}"))
        elif attribution != turn.role.lower():
            v.append(Violation("attribution_mismatch", path, f"attribution {attribution!r} but {ref} is spoken by {turn.role}"))
    if turn.role == "COMPANION" and attribution != "companion" and kind != KIND_REJECTED:
        v.append(Violation("companion_as_fact", path, f"{ref} is a COMPANION statement and must carry attribution 'companion'"))

    if kind != KIND_REJECTED:
        for sentence in split_sentences(turn.text):
            if is_thinking_aloud(sentence.text) and (text in sentence.text or sentence.text in text):
                v.append(Violation("thinking_aloud_as_fact", path, "span comes from clinician thinking-aloud; use kind 'considered_and_rejected'"))
                break

    if FAMILY_WORD_RE.search(value) and not FAMILY_WORD_RE.search(text):
        v.append(Violation("family_word_not_in_span", path, "value names a family member that the span does not"))
    if section not in ("family_history", "social_history") and turn.role in ("PATIENT", "COMPANION") \
            and FAMILY_WORD_RE.search(text) and section not in CLINICIAN_ONLY_SECTIONS:
        v.append(Violation("family_fact_in_personal_section", path, f"span speaks about a relative but sits in {section}"))

    cert = el.get("certainty")
    if section == "assessment":
        if cert not in CERTAINTIES:
            v.append(Violation("certainty_invalid", path, f"assessment needs certainty in {CERTAINTIES}"))
        if FAMILY_RE.search(value) or FAMILY_RE.search(text):
            v.append(Violation("family_history_in_assessment", path, "family-history content must not appear in assessment"))
    elif cert is not None and cert not in CERTAINTIES:
        v.append(Violation("certainty_invalid", path, f"certainty must be one of {CERTAINTIES}"))
    if section in CLINICIAN_ONLY_SECTIONS and turn.role not in CLINICIAN_ROLES:
        v.append(Violation("not_from_clinician", path, f"{section} must come from a DOCTOR/NURSE line, but {ref} is {turn.role}"))
    if section == "review_of_systems":
        if not _NEG_START.match(value):
            v.append(Violation("ros_not_negative", path, "review_of_systems records only negatives (value must start with No/Not/Denies)"))
        elif not _asked_by_doctor(transcript, turn, value):
            v.append(Violation("ros_not_asked", path, "the doctor did not ask about this before it was answered"))
    return v


def validate_note(note: Any, transcript: Transcript) -> list[Violation]:
    violations: list[Violation] = []
    if not isinstance(note, dict):
        return [Violation("structure", "$", "note must be a JSON object")]
    for path, s in _walk_strings(note):
        for name, code in find_code_like(s):
            violations.append(Violation("code_like_string", path, f"{name}-style code {code!r} is not allowed in a note"))
    missing, extra = [s for s in SECTIONS if s not in note], [k for k in note if k not in SECTIONS]
    if missing:
        violations.append(Violation("structure", "$", f"missing section(s): {missing}"))
    if extra:
        violations.append(Violation("structure", "$", f"unexpected top-level key(s): {extra}"))

    family_keys: set[tuple[str, str]] = set()
    fam = note.get("family_history")
    if isinstance(fam, list):
        family_keys = {(e["span"]["ref"], e["span"]["text"]) for e in fam
                       if isinstance(e, dict) and isinstance(e.get("span"), dict) and "ref" in e["span"] and "text" in e["span"]}
    for section in SECTIONS:
        if section not in note:
            continue
        body = note[section]
        if isinstance(body, str):
            if body != NOT_STATED:
                violations.append(Violation("not_stated_literal", f"$.{section}", f"a string section must be exactly {NOT_STATED!r}"))
            continue
        if not isinstance(body, list):
            violations.append(Violation("structure", f"$.{section}", "section must be NOT_STATED or a list of elements"))
            continue
        if not body:
            violations.append(Violation("not_stated_literal", f"$.{section}", f"an empty section must be the literal {NOT_STATED!r}, not []"))
        for i, el in enumerate(body):
            path = f"$.{section}[{i}]"
            violations.extend(check_element(section, el, path, transcript, scan_codes=False))
            if section == "assessment" and isinstance(el, dict) and isinstance(el.get("span"), dict):
                if (el["span"].get("ref"), el["span"].get("text")) in family_keys:
                    violations.append(Violation("family_history_in_assessment", path, "this element duplicates a family_history span"))
    return violations
