# Test cases for the speech evaluation metrics.
import asyncio
import json

import pytest

from app.services.speech_eval.align import align, edit_distance
from app.services.speech_eval.langid import tag
from app.services.speech_eval.metrics import evaluate_texts
from app.services.speech_eval.normaliser import normalise_words
from tests.conftest import ROOT


def norm(text, level="full"):
    return [t for t, _ in normalise_words(text, level)]


@pytest.mark.parametrize("a,b", [
    ("BP is 128 over 82", "BP is 128/82"), ("twenty nineteen", "2019"), ("H. pylori", "H pylori"),
    ("four weeks", "4 weeks"), ("20 milligrams", "20 mg"), ("thirty six point eight", "36.8"), ("94 percent", "94%"),
    ("Nimekua", "nimekuwa"), ("Hello, DOCTOR!", "hello doctor"), ("one hundred and twenty eight", "128"),
])
def test_normaliser_equivalences(a, b):
    assert norm(a) == norm(b)


def test_normaliser_keeps_real_errors_distinct():
    assert norm("diclofenac") != norm("diclofenak")      # a misspelled drug is a clinical error
    assert norm("pulse 76") != norm("pulse 67")
    assert norm("rash") != norm("rush")
    assert norm("No vomiting") != norm("vomiting")


def test_normaliser_levels_differ():
    assert norm("Nimekua", "numbers") != norm("Nimekuwa", "numbers")
    assert norm("twenty nineteen", "basic") == ["twenty", "nineteen"]


def test_alignment_and_edit_distance():
    ops = align(["a", "b", "c"], ["a", "x", "c", "d"])
    assert [o[0] for o in ops] == ["match", "sub", "match", "ins"]
    assert align(["a", "b"], ["b"])[0][0] == "del"
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3


def test_language_tags():
    assert tag("nimekuwa") == "sw" and tag("tumbo") == "sw" and tag("weeks") == "en"
    assert tag("76") == "other" and tag("diclofenac") == "other" and tag("h") == "other"


@pytest.fixture(scope="module")
def metrics(t1_text_module=None):
    ref = (ROOT / "data" / "transcript_01.txt").read_text()
    hyp = (ROOT / "data" / "hyp_01.txt").read_text()
    return asyncio.run(evaluate_texts(ref, hyp))


def test_pair_catches_the_clinical_errors(metrics):
    clinical = {(e["ref"], e["hyp"]) for e in metrics["errors"] if e["clinical"]}
    assert {("76", "67"), ("rash", "rush"), ("diclofenac", "diclofenak")} <= clinical
    assert metrics["clinical_token_error_rate"] > metrics["wer_overall"]
    assert metrics["role_accuracy"] == 1.0 and metrics["normaliser_version"]


def test_required_keys_and_ranges(metrics):
    for k in ("normaliser_version", "wer_overall", "wer_en", "wer_sw", "cer_overall", "clinical_token_error_rate", "role_accuracy",
              "errors", "language_tagging_method", "clinical_token_rule"):
        assert k in metrics
    assert all(0 <= metrics[k] <= 1 for k in ("wer_overall", "wer_en", "wer_sw", "cer_overall", "role_accuracy"))
    for e in metrics["errors"]:
        assert {"ref", "hyp", "type", "lang", "clinical"} <= set(e) and e["type"] in ("sub", "del", "ins") and e["lang"] in ("en", "sw", "other")


def test_committed_metrics_reproduce_exactly(metrics):
    assert json.loads((ROOT / "outputs" / "metrics_01.json").read_text()) == json.loads(json.dumps(metrics))
