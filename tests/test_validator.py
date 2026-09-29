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

