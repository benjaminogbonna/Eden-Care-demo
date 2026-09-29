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

