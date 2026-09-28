# Tests for the extraction service, ensuring that notes are correctly extracted from transcripts and validated.
import asyncio
import json

import pytest

from app.core.transcript import parse_transcript
from app.domain.note import SECTIONS
from app.services.extraction.service import extract_note
from app.services.validation import validate_note
from tests.conftest import ROOT


def values(note, section):
    return [e["value"] for e in note[section]] if note[section] != "NOT_STATED" else []


def test_all_thirteen_sections_and_valid(note1, transcript1):
    assert list(note1) == list(SECTIONS)
    assert validate_note(note1, transcript1) == []


def test_not_stated_where_transcript_is_silent(note1):
    assert note1["past_medical_history"] == "NOT_STATED"


@pytest.mark.parametrize("section,needle", [
    ("chief_complaint", "3 weeks"), ("history_of_presenting_illness", "Worse especially at night"),
    ("history_of_presenting_illness", "after tea"), ("past_surgical_history", "2019"), ("medication_history", "Diclofenac"),
    ("medication_history", "Panadol"), ("allergies", "rash"), ("family_history", "Father had ulcers"), ("social_history", "Does not smoke"),
    ("social_history", "Alcohol"), ("vitals", "BP 128/82"), ("vitals", "Pulse 76"), ("vitals", "Temperature 36.8"),
    ("examination", "No guarding"), ("plan", "Start omeprazole 20 mg once daily for 4 weeks"), ("plan", "H. pylori stool antigen"),
    ("plan", "Come back in 2 weeks"),
])
def test_expected_facts(note1, section, needle):
    assert any(needle in v for v in values(note1, section)), values(note1, section)


def test_review_of_systems_only_what_was_asked(note1):
    ros = values(note1, "review_of_systems")
    assert len(ros) == 3 and all(v.startswith("No ") for v in ros)
    joined = " ".join(ros).lower()
    assert "vomiting" in joined and "stool" in joined and "weight" in joined
    assert "fever" not in joined and "cough" not in joined


def test_assessment_certainty_from_wording(note1):
    (gastritis,) = note1["assessment"]
    assert gastritis["certainty"] == "probable" and "Gastritis" in gastritis["value"]
    assert all("ulcer" not in e["value"].lower() for e in note1["assessment"]), "family history must not leak into assessment"


def test_no_codes_anywhere(note1):
    from app.core.patterns import find_code_like
    assert find_code_like(json.dumps(note1)) == []


def test_every_span_is_verbatim(note1, transcript1):
    for section in SECTIONS:
        for e in (note1[section] if note1[section] != "NOT_STATED" else []):
            assert any(e["span"]["text"] in t.text for t in transcript1.lines_for(e["span"]["ref"]))


def test_committed_outputs_match_fresh_offline_run(t1_text):
    committed = json.loads((ROOT / "outputs" / "note.json").read_text())
    assert committed == asyncio.run(extract_note(t1_text, engine="rules")).note


# the companion / thinking-aloud transcript (own test data)
@pytest.fixture()
def companion_note(companion_text):
    return asyncio.run(extract_note(companion_text, engine="rules")).note


def test_companion_never_a_plain_patient_fact(companion_note, companion_text):
    t = parse_transcript(companion_text)
    n = 0
    for section in SECTIONS:
        for e in (companion_note[section] if companion_note[section] != "NOT_STATED" else []):
            turn = t.lines_for(e["span"]["ref"])[0]
            if turn.role == "COMPANION":
                n += 1
                assert e["attribution"] == "companion"
    assert n >= 3


def test_companion_opinion_not_recorded(companion_note):
    blob = json.dumps(companion_note)
    assert "[01:40]" not in blob, "companion's TB opinion must not be recorded at all"


def test_thinking_aloud_only_as_considered_and_rejected(companion_note):
    tb = [e for e in companion_note["assessment"] if "TB" in e["value"]]
    assert tb and all(e["kind"] == "considered_and_rejected" for e in tb)
    pneumonia = [e for e in companion_note["assessment"] if "Pneumonia" in e["value"]]
    assert pneumonia and pneumonia[0]["certainty"] == "probable" and "kind" not in pneumonia[0]


def test_nurse_vitals_attributed_and_dictated_numbers_kept(companion_note):
    vit = {e["value"]: e["attribution"] for e in companion_note["vitals"]}
    assert vit["BP 150/90"] == "nurse" and vit["Temperature 38.2"] == "nurse" and vit["Oxygen saturation 94%"] == "nurse"


def test_swahili_numeral_and_spoken_number_dose(companion_note):
    meds = values(companion_note, "medication_history")
    assert "Metformin, 500 mg twice daily" in meds
    assert "Start amoxicillin 500 mg 3 times daily for 5 days" in values(companion_note, "plan")


def test_companion_note_validates(companion_note, companion_text):
    assert validate_note(companion_note, parse_transcript(companion_text)) == []


def test_thinking_aloud_drug_never_becomes_plan():
    text = ("[00:00] DOCTOR: Any allergies?\n[00:03] PATIENT: Penicillin. Rash.\n"
            "[00:08] DOCTOR: I was thinking of amoxicillin, but no, not with that allergy. We will give azithromycin.\n")
    note = asyncio.run(extract_note(text, engine="rules")).note
    plain = [e for e in (note["plan"] if note["plan"] != "NOT_STATED" else []) if "kind" not in e]
    assert not any("amoxicillin" in e["value"].lower() for e in plain)
    assert any("azithromycin" in e["value"].lower() for e in plain)
