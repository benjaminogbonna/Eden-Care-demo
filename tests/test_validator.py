# Test cases for the note validator. 
import pytest

from app.services.validation import validate_note


def codes(note, transcript):
    return {v.code for v in validate_note(note, transcript)}


def test_true_note_accepted(note1, transcript1):
    assert validate_note(note1, transcript1) == []


@pytest.mark.parametrize("section,idx,new_value", [
    ("vitals", 1, "Pulse 67"), ("vitals", 0, "BP 182/82"), ("vitals", 2, "Temperature 38.6"),
    ("plan", 1, "Start omeprazole 200 mg once daily for 4 weeks"), ("plan", 1, "Start omeprazole 20 mg once daily for 8 weeks"),
    ("chief_complaint", 0, "Abdominal pain for about 5 weeks, upper abdomen, burning"),
    ("past_surgical_history", 0, "Appendicectomy (appendix), 2018, at the county hospital"),
])
def test_changed_number_rejected(note1, transcript1, tampered, section, idx, new_value):
    bad = tampered(note1, lambda n: n[section][idx].update(value=new_value))
    assert "number_not_in_span" in codes(bad, transcript1)


def test_number_added_where_span_has_none_rejected(note1, transcript1, tampered):
    bad = tampered(note1, lambda n: n["examination"][0].update(value="Abdomen soft, 3 cm mass"))
    assert "number_not_in_span" in codes(bad, transcript1)


@pytest.mark.parametrize("value,span", [("Duration 3 weeks", "for about three weeks"), ("Duration three weeks", "for about 3 weeks"),
                                        ("2019", "twenty nineteen"), ("twenty nineteen", "2019")])
def test_spoken_and_digit_forms_both_directions(value, span):
    from app.core.transcript import parse_transcript
    t = parse_transcript(f"[00:00] PATIENT: I had it {span} ok")
    note = {s: "NOT_STATED" for s in __import__("app.domain.note", fromlist=["SECTIONS"]).SECTIONS}
    note["history_of_presenting_illness"] = [{"value": value, "span": {"ref": "[00:00]", "text": span}, "confidence": 1.0}]
    assert validate_note(note, t) == []


def test_changed_spelled_number_rejected():
    from app.core.transcript import parse_transcript
    from app.domain.note import SECTIONS
    t = parse_transcript("[00:00] PATIENT: pain for three weeks")
    note = {s: "NOT_STATED" for s in SECTIONS}
    note["history_of_presenting_illness"] = [{"value": "Pain for 4 weeks", "span": {"ref": "[00:00]", "text": "for three weeks"}, "confidence": 1.0}]
    assert "number_not_in_span" in codes(note, t)


@pytest.mark.parametrize("bad_value", ["Diclofenac M01AB05", "Gastritis K29.7", "LAB-EC-0412", "ALG-EC-0003 penicillin", "code J01CA04"])
def test_code_like_string_rejected_anywhere(note1, transcript1, tampered, bad_value):
    bad = tampered(note1, lambda n: n["plan"][0].update(value=bad_value))
    assert "code_like_string" in codes(bad, transcript1)


def test_code_in_key_or_entity_rejected(note1, transcript1, tampered):
    assert "code_like_string" in codes(tampered(note1, lambda n: n["plan"][0].update(entity="K29.7")), transcript1)
    assert "code_like_string" in codes(tampered(note1, lambda n: n["plan"][0].update({"K29": 1})), transcript1)
    assert "code_like_string" in codes(tampered(note1, lambda n: n["plan"][0]["span"].update(text="M01AB05")), transcript1)


def test_family_history_moved_to_assessment_rejected(note1, transcript1, tampered):
    def move(n):
        fam = n["family_history"][0]
        n["assessment"].append({**fam, "certainty": "confirmed"})
    assert "family_history_in_assessment" in codes(tampered(note1, move), transcript1)


def test_family_word_smuggled_into_assessment_value(note1, transcript1, tampered):
    bad = tampered(note1, lambda n: n["assessment"][0].update(value="Family history of gastric ulcer (father)"))
    assert {"family_history_in_assessment", "family_word_not_in_span"} <= codes(bad, transcript1)


def test_family_fact_in_personal_section_rejected(note1, transcript1, tampered):
    bad = tampered(note1, lambda n: n.update(past_medical_history=[dict(n["family_history"][0])]))
    assert "family_fact_in_personal_section" in codes(bad, transcript1)


def test_span_must_be_verbatim(note1, transcript1, tampered):
    bad = tampered(note1, lambda n: n["vitals"][1]["span"].update(text="pulse seventy six"))
    assert "span_not_verbatim" in codes(bad, transcript1)
    bad = tampered(note1, lambda n: n["vitals"][1]["span"].update(ref="[09:99]"))
    assert "span_ref_unknown" in codes(bad, transcript1)


def test_structure_rules(note1, transcript1, tampered):
    assert "structure" in codes(tampered(note1, lambda n: n.pop("plan")), transcript1)
    assert "structure" in codes(tampered(note1, lambda n: n.update(extra=[])), transcript1)
    assert "not_stated_literal" in codes(tampered(note1, lambda n: n.update(past_medical_history=[])), transcript1)
    assert "not_stated_literal" in codes(tampered(note1, lambda n: n.update(past_medical_history="none")), transcript1)
    assert "confidence_range" in codes(tampered(note1, lambda n: n["plan"][0].update(confidence=1.5)), transcript1)
    assert "certainty_invalid" in codes(tampered(note1, lambda n: n["assessment"][0].update(certainty="certain")), transcript1)
    assert "structure" in validate_note([], transcript1)[0].code


def test_companion_as_fact_and_attribution(companion_text, tampered):
    import asyncio
    from app.core.transcript import parse_transcript
    from app.services.extraction.service import extract_note
    note = asyncio.run(extract_note(companion_text, engine="rules")).note
    t = parse_transcript(companion_text)
    def strip(n): del n["history_of_presenting_illness"][0]["attribution"]
    assert "companion_as_fact" in codes(tampered(note, strip), t)
    def relabel(n): n["history_of_presenting_illness"][0]["attribution"] = "patient"
    assert {"companion_as_fact", "attribution_mismatch"} <= codes(tampered(note, relabel), t)


def test_thinking_aloud_as_plain_fact_rejected(companion_text, tampered):
    import asyncio
    from app.core.transcript import parse_transcript
    from app.services.extraction.service import extract_note
    note = asyncio.run(extract_note(companion_text, engine="rules")).note
    t = parse_transcript(companion_text)
    def unmark(n): del n["assessment"][0]["kind"]
    assert "thinking_aloud_as_fact" in codes(tampered(note, unmark), t)


def test_ros_must_be_negative_and_asked(note1, transcript1, tampered):
    assert "ros_not_negative" in codes(tampered(note1, lambda n: n["review_of_systems"][0].update(value="Vomiting")), transcript1)
    bad = tampered(note1, lambda n: n["review_of_systems"].append({"value": "No fever", "span": {"ref": "[00:22]", "text": "No vomiting"}, "confidence": 1.0, "attribution": "patient"}))
    assert "ros_not_asked" in codes(bad, transcript1)


def test_patient_cannot_supply_assessment(note1, transcript1, tampered):
    bad = tampered(note1, lambda n: n["assessment"].append({"value": "Gastritis", "span": {"ref": "[00:04]", "text": "tumbo kuuma"}, "confidence": 1.0, "certainty": "confirmed"}))
    assert "not_from_clinician" in codes(bad, transcript1)
