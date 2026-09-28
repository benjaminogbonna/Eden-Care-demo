import pytest

from app.core.numbers import digitize, numeric_values, parse_number
from app.core.patterns import find_code_like, is_thinking_aloud
from app.core.text import split_sentences
from app.core.transcript import parse_transcript
from app.core.errors import TranscriptError


def test_parse_transcript(t1_text):
    t = parse_transcript(t1_text)
    assert len(t) == 19 and t.turns[0].role == "DOCTOR" and t.turns[1].ref == "[00:04]"
    assert t.lines_for("[01:03]")[0].text.startswith("BP is 128")


def test_continuation_lines_and_errors():
    t = parse_transcript("[00:00] DOCTOR: Hello\nthere\n[00:05] PATIENT: Hi")
    assert t.turns[0].text == "Hello there"
    with pytest.raises(TranscriptError):
        parse_transcript("no timestamps here")
    with pytest.raises(TranscriptError):
        parse_transcript("   \n")


@pytest.mark.parametrize("words,expected", [
    ("twenty nineteen", "2019"), ("nineteen ninety five", "1995"), ("two thousand and nineteen", "2019"),
    ("one hundred and twenty eight", "128"), ("thirty six point eight", "36.8"), ("twenty one", "21"), ("five hundred", "500"),
])
def test_spoken_numbers(words, expected):
    assert parse_number(words.split(), 0)[0] == expected


def test_numeric_values_both_forms():
    assert numeric_values("one tablet, about three weeks, 36.8, twenty nineteen, mara mbili") == {"1", "3", "36.8", "2019", "2"}


def test_digitize_only_before_units():
    assert digitize("four weeks and one of them, 20 milligrams, 128 over 82") == "4 weeks and one of them, 20 mg, 128/82"


def test_sentence_split_keeps_initials():
    assert [s.text for s in split_sentences("Check an H. pylori test. Temp 36.8. No. Fine!")] == ["Check an H. pylori test.", "Temp 36.8.", "No.", "Fine!"]


def test_code_patterns_match_contract():
    assert find_code_like("K29.7") and find_code_like("M01AB05") and find_code_like("LAB-EC-0412")
    assert not find_code_like("128/82 in 2019, 20 mg, 36.8")


def test_thinking_aloud_detection():
    assert is_thinking_aloud("I was wondering whether this could be TB, but let's not go there yet.")
    assert not is_thinking_aloud("I think this is most likely gastritis.")
