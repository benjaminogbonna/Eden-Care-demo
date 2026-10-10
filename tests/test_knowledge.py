# Tests for knowledge building and verification
import pytest

from app.core.errors import InputError
from app.services.knowledge import build_knowledge, verify_knowledge


def test_rows_and_grounding(guideline_text):
    k = build_knowledge(guideline_text)
    types = [r["type"] for r in k["rows"]]
    assert types.count("red_flag") == 6 and "test_constraint" in types and types.count("drug_class_rule") >= 3
    assert verify_knowledge(k, guideline_text) == []
    for r in k["rows"]:
        assert (r["source"], r["section"], r["page"]) == ("Synthetic Gastroenterology Guideline", "4.2", 17)
        assert r["quote"] in guideline_text


def test_test_constraint_and_ppi_duration(guideline_text):
    k = build_knowledge(guideline_text)
    tc = next(r for r in k["rows"] if r["type"] == "test_constraint")
    assert tc["fields"]["preceding_period"] == "two weeks" and "sensitivity" in tc["fields"]["reason"]
    ppi = next(r for r in k["rows"] if r["type"] == "drug_class_rule" and "duration_weeks" in r["fields"])
    assert ppi["fields"]["duration_weeks"] == {"min": 4, "max": 8}
    assert any(r["fields"].get("age_over") == 55 for r in k["rows"] if r["type"] == "red_flag")


def test_not_in_corpus_and_prose(guideline_text):
    k = build_knowledge(guideline_text)
    assert len(k["not_in_corpus"]) >= 2
    joined = " ".join(k["not_in_corpus"]).lower()
    assert "eradication" in joined and "dose" in joined
    assert len(k["prose"].split()) < 200


def test_gap_disappears_when_source_covers_it(guideline_text):
    extended = guideline_text + " Eradication therapy: amoxicillin plus clarithromycin."
    assert not any("eradication" in g.lower() for g in build_knowledge(extended)["not_in_corpus"])


def test_tampered_quote_detected(guideline_text):
    k = build_knowledge(guideline_text)
    k["rows"][0]["quote"] += " and always give antibiotics"
    assert verify_knowledge(k, guideline_text)


def test_missing_header_or_rules_rejected():
    with pytest.raises(InputError, match="header"):
        build_knowledge("Just some text with no header.")
    with pytest.raises(InputError, match="no recognisable"):
        build_knowledge("SOURCE: Some Book, 1st edition, Section 1.1, page 2.\nNothing useful here.")
