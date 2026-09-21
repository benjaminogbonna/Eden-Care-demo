"""Deterministic resolver -> resolved.json.
"""
from __future__ import annotations

from typing import Any

from app.services.resolver.matching import normalise, score
from app.services.resolver.register import Register

RESOLVE_AT = 0.80      # top score needed to assign a code
AMBIGUOUS_AT = 0.55    # between this and RESOLVE_AT the item is flagged ambiguous, code stays null
MARGIN = 0.05          # a runner-up closer than this makes the item ambiguous
ALT_FLOOR = 0.40

SECTION_KINDS: dict[str, tuple[str, ...]] = {
    "past_medical_history": ("diagnosis",),
    "past_surgical_history": ("procedure",),
    "medication_history": ("drug",),
    "allergies": ("allergen",),
    "assessment": ("diagnosis",),
    "plan": ("drug", "lab", "procedure"),
}
UNCODED_SECTIONS = {
    "chief_complaint": "symptoms are not in the register",
    "history_of_presenting_illness": "narrative section; not coded",
    "review_of_systems": "negatives are not coded",
    "family_history": "family history is not a patient diagnosis and is never coded",
    "social_history": "not coded",
    "vitals": "measurements are not coded",
    "examination": "findings are not coded",
}
CODE_SYSTEM = {
    "diagnosis": "ICD-10", "drug": "ATC", "lab": "EDEN-CARE-LAB", "procedure": "EDEN-CARE-PGP", "allergen": "EDEN-CARE-ALG",
}
NONE_SYSTEM = "none"


def _candidates(query: str, kinds: tuple[str, ...], register: Register) -> list[dict[str, Any]]:
    q = normalise(query)
    out = []
    for entry in register.entries:
        if entry.kind not in kinds:
            continue
        best = max((score(q, normalise(term)) for term in entry.terms), default=0.0)
        if best >= ALT_FLOOR:
            out.append({"code": entry.code, "name": entry.name, "score": round(best, 3), "kind": entry.kind})
    out.sort(key=lambda c: (-c["score"], c["code"]))
    return out


def _uncoded(el: dict[str, Any], reason: str, system: str = NONE_SYSTEM) -> dict[str, Any]:
    return {**el, "extraction_confidence": el.get("confidence"), "code": None, "code_system": system,
            "confidence": 0.0, "alternatives": [], "status": "unresolved", "reason": reason}


def resolve_element(section: str, el: dict[str, Any], register: Register) -> dict[str, Any]:
    if section in UNCODED_SECTIONS:
        return _uncoded(el, UNCODED_SECTIONS[section])
    kinds = SECTION_KINDS[section]
    system = CODE_SYSTEM[kinds[0]]
    if el.get("kind") == "considered_and_rejected":
        return _uncoded(el, "considered and rejected by the clinician; never coded", system)

    query = el.get("entity") or el.get("value") or ""
    cands = _candidates(query, kinds, register)
    out = {**el, "extraction_confidence": el.get("confidence")}
    if el.get("attribution") == "companion":
        out["requires_confirmation"] = True
    if not cands or cands[0]["score"] < AMBIGUOUS_AT:
        out.update(code=None, code_system=system, confidence=cands[0]["score"] if cands else 0.0,
                   alternatives=[_alt(c) for c in cands], status="unresolved",
                   reason="no register entry is close enough; left uncoded")
        return out
    top = cands[0]
    tie = len(cands) > 1 and top["score"] - cands[1]["score"] < MARGIN and cands[1]["score"] >= AMBIGUOUS_AT
    if top["score"] >= RESOLVE_AT and not tie:
        out.update(code=top["code"], code_system=CODE_SYSTEM[top["kind"]], confidence=top["score"],
                   alternatives=[_alt(c) for c in cands[1:]], status="resolved")
    else:
        out.update(code=None, code_system=system, confidence=top["score"],
                   alternatives=[_alt(c) for c in cands if c["score"] >= AMBIGUOUS_AT], status="ambiguous",
                   reason="more than one plausible register entry, or only a partial match; left uncoded")
    return out


def _alt(c: dict[str, Any]) -> dict[str, Any]:
    return {"code": c["code"], "name": c["name"], "score": c["score"]}


def resolve_note(note: dict[str, Any], register: Register) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for section, body in note.items():
        if isinstance(body, list):
            resolved[section] = [resolve_element(section, el, register) for el in body]
        else:
            resolved[section] = body
    return resolved
