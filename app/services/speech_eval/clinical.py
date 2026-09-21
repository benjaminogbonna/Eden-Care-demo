"""Clinical token classification for the clinical-token error rate.
"""
from __future__ import annotations

from app.core.numbers import SWAHILI_NUMBERS
from app.services.speech_eval.langid import clinical_terms

RULE = (
    "clinical reference tokens = negations, drug/lab names, allergy and reaction words, numbers (doses, vitals, years, "
    "durations), dose units and frequencies, vital-sign names, duration/date words; an error is clinical if the reference "
    "token is clinical (sub/del), if the hypothesis token is a negation or number (sub), or if the inserted token is "
    "clinical (ins); CTER = clinical errors / clinical reference tokens"
)
NEGATIONS = frozenset("no not none never without nothing hapana sina hakuna hana hajawahi si denies nil cannot".split())
ALLERGY = frozenset("allergy allergic allergies mzio rash hives itching swelling anaphylaxis wheezing".split())
DOSE_UNITS = frozenset("mg mcg g ml kg iu tablet tablets tab tabs capsule capsules drops puffs once twice daily bd tds od qid hourly".split())
VITALS = frozenset("bp pulse temperature temp sats saturation respiratory rate mmhg bpm glucose heart".split())
DATES = frozenset(
    "week weeks day days month months year years yesterday ago wiki siku mwezi miezi mwaka miaka jana juzi "
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday".split()
)


def classify(tokens: list[str]) -> list[str | None]:
    """Class of each token (None = not clinical), using one token of left context for 'mara <numeral>'."""
    drugs = clinical_terms()
    out: list[str | None] = []
    for i, t in enumerate(tokens):
        if t in NEGATIONS or t.endswith("n't") or t.startswith("haku"):
            out.append("negation")
        elif t in drugs:
            out.append("drug")
        elif t in ALLERGY:
            out.append("allergy")
        elif t[0].isdigit():
            out.append("number")
        elif t in DOSE_UNITS:
            out.append("dose_unit")
        elif t in VITALS:
            out.append("vital")
        elif t in DATES:
            out.append("date")
        elif i > 0 and tokens[i - 1] == "mara" and t in SWAHILI_NUMBERS:
            out.append("frequency")
        else:
            out.append(None)
    return out
