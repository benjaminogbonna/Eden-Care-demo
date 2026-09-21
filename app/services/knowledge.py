"""Grounded knowledge rows from a guideline excerpt
"""
from __future__ import annotations

import re
from typing import Any

from app.core.numbers import parse_number
from app.core.text import split_sentences
from app.core.errors import InputError

HEADER_RE = re.compile(
    r"^\s*SOURCE:\s*(?P<title>.+?)(?:,\s*\d+(?:st|nd|rd|th)\s+edition)?,\s*Section\s+(?P<section>\d+(?:\.\d+)*),\s*page\s+(?P<page>\d+)\s*\.?\s*$",
    re.I | re.M,
)

GAP_CHECKS: list[tuple[str, str]] = [
    ("Which PPI to prescribe and its standard dose in milligrams (the source says only 'standard dose')", r"\bmg\b|milligram|omeprazole|esomeprazole|pantoprazole"),
    ("H. pylori eradication regimen and antibiotic choice, including alternatives for a penicillin-allergic patient", r"eradicat|amoxicillin|clarithromycin|metronidazole|antibiotic"),
    ("A safe analgesic alternative once the NSAID is stopped (for example for the condition the NSAID was treating)", r"paracetamol|acetaminophen|analgesi|alternative"),
    ("Follow-up interval, when to review, and whether to repeat testing after treatment", r"follow[- ]?up|review after|repeat|test of cure"),
    ("What to do when H. pylori testing is needed but a PPI cannot safely be withheld for two weeks", r"withhold|washout|serolog|breath test|endoscop"),
    ("Definitions or thresholds for 'unintentional weight loss' and 'persistent vomiting'", r"\bkg\b|\d+\s*%|percent|weeks of vomiting"),
    ("Endoscopy indications beyond the listed alarm features", r"endoscop|gastroscop"),
]


def _weeks_range(sentence: str) -> dict[str, int] | None:
    m = re.search(r"\b(\w+) to (\w+) weeks\b", sentence, re.I)
    if not m:
        return None
    vals = []
    for w in (m.group(1), m.group(2)):
        parsed = int(w) if w.isdigit() else (int(parse_number([w.lower()], 0)[0]) if parse_number([w.lower()], 0) else None)
        if parsed is None:
            return None
        vals.append(parsed)
    return {"min": vals[0], "max": vals[1]}


def build_knowledge(source_text: str, *, source_name: str = "guideline") -> dict[str, Any]:
    header = HEADER_RE.search(source_text)
    if not header:
        raise InputError(
            f"{source_name}: could not find a header like 'SOURCE: <title>, Section <n.n>, page <n>.'; "
            "every knowledge row needs a source, section and page"
        )
    title, section, page = header.group("title").strip(), header.group("section"), int(header.group("page"))
    body = source_text[header.end():]
    sentences = [s.text for s in split_sentences(body)]

    rows: list[dict[str, Any]] = []

    def add(rtype: str, fields: dict[str, Any], quote: str) -> None:
        if quote not in source_text:  # invariant: never emit a quote that is not in the source
            raise AssertionError(f"quote not found in source: {quote!r}")
        rows.append({"type": rtype, "fields": fields, "source": title, "section": section, "page": page, "quote": quote})

    for s in sentences:
        low = s.lower()
        if "nsaid" in low and re.search(r"should have the nsaid stopped|\bstop(?:ped)?\b", low) and "co-prescription" not in low:
            add("drug_class_rule", {
                "drug_class": "NSAID", "action": "stop where possible",
                "applies_when": "epigastric burning pain in a patient taking an NSAID",
            }, s)
        if "proton pump inhibitor" in low and "recommended" in low:
            fields: dict[str, Any] = {"drug_class": "proton pump inhibitor (PPI)", "action": "recommended", "dose": "standard dose"}
            weeks = _weeks_range(s)
            if weeks:
                fields["duration_weeks"] = weeks
            add("drug_class_rule", fields, s)
        if "co-prescription" in low or "coprescription" in low:
            add("drug_class_rule", {
                "drug_class": "NSAID with PPI", "action": "PPI co-prescription for gastroprotection",
                "applies_when": "the patient requires ongoing NSAID therapy",
            }, s)
        if low.startswith("test for helicobacter pylori") or ("stool antigen" in low and "should not have taken" in low):
            m = re.search(r"should not have taken an? (?P<drug>[^.;]*?) in the preceding (?P<period>[\w ]+?) as this (?P<why>[^.;]+)", s, re.I)
            fields = {"test": "Helicobacter pylori stool antigen"}
            if m:
                fields.update({"constraint": f"patient should not have taken a {m.group('drug')} in the preceding {m.group('period')}",
                               "excluded_drug": m.group("drug"), "preceding_period": m.group("period"),
                               "reason": m.group("why").strip()})
            add("test_constraint", fields, s)
        if low.startswith("alarm features"):
            _, _, tail = s.partition(":")
            for feature in [f.strip().rstrip(".") for f in tail.split(",") if f.strip()]:
                fields = {"feature": feature, "action": "urgent referral"}
                age = re.search(r"age over (\d+)", feature, re.I)
                if age:
                    fields["age_over"] = int(age.group(1))
                add("red_flag", fields, s)

    if not rows:
        raise InputError(f"{source_name}: no recognisable guideline rules (drug-class rule, red flag, test constraint) found")

    gaps = [q for q, present in GAP_CHECKS if not re.search(present, source_text, re.I)]
    return {"rows": rows, "not_in_corpus": gaps, "prose": _prose(rows, gaps, title, section, page)}


def _prose(rows: list[dict], gaps: list[str], title: str, section: str, page: int) -> str:
    flags = [r["fields"]["feature"] for r in rows if r["type"] == "red_flag"]
    ppi = next((r for r in rows if r["type"] == "drug_class_rule" and "PPI" in r["fields"]["drug_class"] and "duration_weeks" in r["fields"]), None)
    test = next((r for r in rows if r["type"] == "test_constraint"), None)
    parts = [f"{title}, section {section}, page {page}: for epigastric burning pain in a patient taking an NSAID, stop the NSAID where possible"]
    if ppi:
        w = ppi["fields"]["duration_weeks"]
        parts[0] += f" and give a standard-dose PPI for {w['min']} to {w['max']} weeks; patients who need ongoing NSAID therapy should also receive a PPI."
    else:
        parts[0] += "."
    if test and "preceding_period" in test["fields"]:
        parts.append(f"Test for H. pylori with a stool antigen test, but not if the patient took a PPI in the preceding {test['fields']['preceding_period']}, because that reduces sensitivity.")
    if flags:
        parts.append("Urgent referral for: " + "; ".join(flags) + ".")
    if gaps:
        parts.append(f"The source does not say: {len(gaps)} items a clinician would want, including the PPI name and dose in mg and the H. pylori eradication regimen.")
    return " ".join(parts)


def verify_knowledge(knowledge: dict[str, Any], source_text: str) -> list[str]:
    """Return a list of problems; empty means every row quote is verbatim in the source."""
    problems = []
    for i, row in enumerate(knowledge.get("rows", [])):
        if row.get("quote") not in source_text:
            problems.append(f"rows[{i}] quote is not in the source")
        for k in ("type", "fields", "source", "section", "page", "quote"):
            if k not in row:
                problems.append(f"rows[{i}] missing {k}")
    return problems
