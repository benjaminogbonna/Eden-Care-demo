# Tests for the resolver module. 
import re
from pathlib import Path

import pytest

from app.core.errors import NoteFormatError, RegisterError
from app.services.resolution import resolve_checked
from app.services.resolver import load_register, parse_register, resolve_note
from tests.conftest import ROOT


@pytest.fixture()
def register(register_path):
    return load_register(register_path)


def el(value, **kw):
    return {"value": value, "span": {"ref": "[00:00]", "text": value}, "confidence": 1.0, **kw}


def one(section, element, register):
    return resolve_note({section: [element]}, register)[section][0]


def by_value(resolved, section, needle):
    return next(e for e in resolved[section] if needle in e["value"])


def test_expected_codes_for_transcript_01(note1, register):
    r = resolve_note(note1, register)
    assert by_value(r, "medication_history", "Diclofenac")["code"] == "M01AB05"
    assert by_value(r, "medication_history", "Panadol")["code"] == "N02BE01"
    assert by_value(r, "allergies", "Penicillin")["code"] == "ALG-EC-0003"
    assert by_value(r, "assessment", "Gastritis")["code"] == "K29.7"
    assert by_value(r, "plan", "omeprazole")["code"] == "A02BC01"
    assert by_value(r, "plan", "H. pylori")["code"] == "LAB-EC-0412"
    assert by_value(r, "plan", "Stop")["code"] == "M01AB05"


def test_appendix_resolves_to_procedure_not_diagnosis(note1, register):
    e = by_value(resolve_note(note1, register), "past_surgical_history", "Appendicectomy")
    assert e["code"] == "PGP-EC-0330" and e["code_system"] == "EDEN-CARE-PGP" and e["status"] == "resolved"
    # even the bare word "appendix" cannot reach the diagnosis in a surgical-history section
    assert one("past_surgical_history", el("Appendix surgery, 2019"), register)["code"] == "PGP-EC-0330"


def test_family_history_never_coded(note1, register):
    e = resolve_note(note1, register)["family_history"][0]
    assert e["code"] is None and e["status"] == "unresolved" and "family" in e["reason"]


def test_same_word_different_section_different_kind(register):
    assert one("assessment", el("Appendicitis"), register)["code"] not in (None, "PGP-EC-0330")


def test_unmatched_left_uncoded_not_forced(register):
    for value, section in (("Metformin 500 mg", "medication_history"), ("Cisplatin", "plan"), ("Hypertension", "past_medical_history")):
        e = one(section, el(value), register)
        assert e["code"] is None and e["status"] == "unresolved", value
    assert one("plan", el("Come back in 2 weeks"), register)["status"] == "unresolved"


@pytest.mark.parametrize("value,code", [("Diclofenak", "M01AB05"), ("voltaren gel", "M01AB05"), ("amoxycillin", "J01CA04"), ("Omez", "A02BC01")])
def test_near_misses_and_brands(register, value, code):
    e = one("medication_history", el(value), register)
    assert e["code"] == code and e["status"] == "resolved"


def test_partial_match_is_ambiguous_with_null_code(register):
    e = one("past_medical_history", el("ulcer"), register)
    assert e["code"] is None and e["status"] == "ambiguous" and e["alternatives"]


def test_rejected_and_companion_handling(register):
    rej = one("assessment", el("Gastritis considered, not pursued", kind="considered_and_rejected"), register)
    assert rej["code"] is None and "rejected" in rej["reason"]
    comp = one("medication_history", el("Diclofenac", attribution="companion"), register)
    assert comp["code"] == "M01AB05" and comp["requires_confirmation"] is True


def test_shape_and_extraction_confidence_preserved(note1, register):
    r = resolve_note(note1, register)
    assert list(r) == list(note1)
    e = r["plan"][0]
    assert e["extraction_confidence"] == note1["plan"][0]["confidence"] and {"code", "code_system", "confidence", "alternatives", "status"} <= set(e)
    assert r["past_medical_history"] == "NOT_STATED"


def test_resolve_checked_rejects_bad_notes(register):
    for bad in ([], {"nonsense": []}, {"plan": "oops"}, {"plan": [3]}):
        with pytest.raises(NoteFormatError):
            resolve_checked(bad, register)


# register loading: every failure names the file and (where relevant) the line
GOOD = "kind,code,name,synonyms\ndrug,A1,Alphadrug,alpha;alphax\n"


@pytest.mark.parametrize("text,fragment", [
    ("", "empty"),
    ("kind,code,name\ndrug,A1,Alpha\n", "missing required column"),
    ("kind,code,name,synonyms\n", "no entries"),
    (GOOD + "drug,A2,Beta\n", "line 3: expected 4 fields but found 3"),
    (GOOD + 'drug,A2,"Beta,syn\n', "unparseable"),
    (GOOD + "drug,A1,Alpha again,x\n", "duplicated"),
    (GOOD + "lab,A1,Some lab,x\n", "two different kinds"),
    (GOOD + "poison,A2,Beta,x\n", "unknown kind"),
    (GOOD + "drug,,Beta,x\n", "non-empty"),
])
def test_register_errors_are_specific(text, fragment):
    with pytest.raises(RegisterError) as e:
        parse_register(text, "my.csv")
    assert "my.csv" in e.value.message and fragment in e.value.message


def test_missing_register_file(tmp_path):
    with pytest.raises(RegisterError, match="not found"):
        load_register(tmp_path / "nope.csv")


def test_bundled_register_loads(register):
    assert len(register.entries) == 12


def test_resolver_package_has_no_network_or_model_calls():
    """The resolver must be a pure function of (note, register)."""
    banned = re.compile(r"\b(?:import|from)\s+(?:requests|httpx|urllib|urllib3|aiohttp|socket|http|openai|anthropic|google|genai|litellm|langchain)\b", re.I)
    words = re.compile(r"openai|anthropic|gemini|genai|urlopen|socket|httpx|requests\.|aiohttp", re.I)
    for path in (ROOT / "app" / "services" / "resolver").glob("*.py"):
        src = path.read_text()
        assert not banned.search(src), path
        assert not words.search(src), path
        assert "app.services.extraction" not in src and "app.core.config" not in src, f"{path} must not depend on the extraction layer or settings"
