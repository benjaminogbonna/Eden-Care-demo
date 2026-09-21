"""Small bilingual clinical lexicon for the rules engine.
"""
from __future__ import annotations

import re

from app.core.numbers import digitize

# gloss table
_SW_NUM = {"moja": "1", "mbili": "2", "tatu": "3", "nne": "4", "tano": "5", "sita": "6", "saba": "7", "nane": "8", "tisa": "9", "kumi": "10"}
_FREQ_WORD = {"moja": "once daily", "mbili": "twice daily", "tatu": "3 times daily", "nne": "4 times daily"}
_FREQ_RE = re.compile(r"\bmara\s+(moja|mbili|tatu|nne)\s+kwa\s+siku\b", re.I)
_DUR_RE = re.compile(r"\bkwa\s+(wiki|siku|miezi|mwezi|mwaka|miaka)\s+(moja|mbili|tatu|nne|tano|sita|saba|nane|tisa|kumi)\b", re.I)
_DUR_UNIT = {"wiki": "weeks", "siku": "days", "miezi": "months", "mwezi": "month", "mwaka": "year", "miaka": "years"}

GLOSS: dict[str, str] = {
    "nimekuwa na": "", "nimekua na": "", "nimekuwa": "", "nimekua": "", "nina": "", "nilikuwa na": "had",
    "alikuwa na": "had", "nimepata": "", "nilipata": "had",
    "tumbo kuuma": "abdominal pain", "tumbo kuwaka": "burning abdominal pain", "tumbo": "stomach",
    "kama kuwaka": "burning", "kuwaka": "burning",
    "usiku sana": "especially at night", "usiku": "at night", "asubuhi": "in the morning",
    "jioni": "in the evening", "mchana": "during the day", "kila siku": "every day", "kwa siku": "daily",
    "kidogo": "a little", "sana": "a lot", "ni sawa": "is fine", "sawa": "fine",
    "pombe": "alcohol", "sigara": "cigarettes",
    "nakohoa": "coughing", "kikohozi": "cough", "homa": "fever", "kutapika": "vomiting", "kuhara": "diarrhoea",
    "maumivu ya kichwa": "headache", "kichwa kuuma": "headache", "kifua kinauma": "chest pain", "kifua": "chest",
    "mgongo": "back", "goti": "knee", "uchovu": "tiredness", "kuchoka": "tiredness", "kizunguzungu": "dizziness",
    "damu": "blood", "dawa ya": "medicine for", "dawa": "medicine", "vidonge": "tablets", "kidonge": "tablet",
    "sijui jina": "name not known", "sijui": "I do not know", "jina": "name",
    "sukari": 'diabetes ("sukari")', "na": "and",
}
_PRESSURE_RE = re.compile(r'(?<!blood )\bpressure\b', re.I)
_GLOSS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(GLOSS, key=len, reverse=True)) + r")\b", re.I
)


def gloss(text: str) -> str:
    """English rendering of a (possibly code-switched) patient sentence for `value`.

    Word-for-word phrase substitution only: it never adds a clinical claim that is not in the words.
    """
    s = text.strip().rstrip(".!?").strip()
    low = s.lower()
    s = re.sub(r"\bhapa juu\b", "upper abdomen" if "tumbo" in low else "upper part", s, flags=re.I)
    s = _FREQ_RE.sub(lambda m: _FREQ_WORD[m.group(1).lower()], s)
    s = _DUR_RE.sub(lambda m: f"for {_SW_NUM[m.group(2).lower()]} {_DUR_UNIT[m.group(1).lower()]}", s)
    s = _PRESSURE_RE.sub('hypertension ("pressure")', s)
    s = _GLOSS_RE.sub(lambda m: GLOSS[m.group(0).lower()], s)
    s = digitize(s)
    s = re.sub(r"^\s*my\s+", "", s, flags=re.I)
    s = re.sub(r"^I do(?: not|n't)\s+", "Does not ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+,", ",", s).strip(" ,;")
    s = re.sub(r"^(?:and|but)\s+", "", s, flags=re.I)
    return s[:1].upper() + s[1:] if s else s


def capitalise(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


# entity lexicons
def _alt(words: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(sorted(words, key=len, reverse=True)) + r")\b", re.I)


DRUG_RE = _alt(
    """diclofenac voltaren paracetamol panadol acetaminophen omeprazole omez amoxicillin amoxil amoxycillin
    ibuprofen brufen aspirin naproxen metformin glibenclamide gliclazide insulin amlodipine nifedipine atenolol
    enalapril lisinopril losartan hydrochlorothiazide furosemide metronidazole flagyl ciprofloxacin ceftriaxone
    azithromycin doxycycline cotrimoxazole septrin artemether lumefantrine coartem artesunate quinine albendazole
    salbutamol prednisolone cetirizine loratadine ranitidine esomeprazole pantoprazole lansoprazole tramadol
    morphine diazepam phenytoin carbamazepine atorvastatin simvastatin warfarin clopidogrel rifampicin isoniazid
    pyrazinamide ethambutol tenofovir lamivudine dolutegravir efavirenz erythromycin cloxacillin penicillin
    gentamicin ors zinc""".split()
)
ALLERGEN_RE = _alt(
    """penicillins? amoxicillin sulfa sulpha sulphonamides? aspirin nsaids? ibuprofen diclofenac latex peanuts?
    eggs? shellfish""".split() + ["bee stings?", "dust"]
)
REACTION_RE = _alt(
    ["rash", "hives", "itching", "swelling", "anaphylaxis", "wheezing", "difficulty breathing",
     "shortness of breath", "vomiting", "kuwasha", "vipele"]
)
LAB_RE = re.compile(
    r"\b(?:h\.?\s*pylori(?:\s+stool)?(?:\s+antigen)?(?:\s+test)?|full blood count|full haemogram|fbc|cbc|"
    r"malaria\s+(?:test|smear|rdt|slide)|blood film|(?:chest\s+)?x-?ray|ultrasound|urinalysis|urine\s+test|"
    r"stool\s+(?:test|analysis)|blood sugar|glucose test|hiv test|sputum(?:\s+test)?|ecg|ct scan|culture)\b",
    re.I,
)

# display name, pattern
DIAGNOSES: list[tuple[str, re.Pattern[str]]] = [
    ("Gastric ulcer", re.compile(r"\b(?:gastric|peptic|stomach)\s+ulcers?\b|\bkidonda cha tumbo\b", re.I)),
    ("Gastritis", re.compile(r"\bgastritis\b", re.I)),
    ("GERD", re.compile(r"\bgerd\b|\bacid reflux\b|\breflux\b", re.I)),
    ("Appendicitis", re.compile(r"\bappendicitis\b", re.I)),
    ("Pneumonia", re.compile(r"\bpneumonia\b|\bchest infection\b", re.I)),
    ("TB", re.compile(r"\btb\b|\btuberculosis\b", re.I)),
    ("Malaria", re.compile(r"\bmalaria\b", re.I)),
    ("Hypertension", re.compile(r"\bhypertension\b|\bhigh blood pressure\b", re.I)),
    ("Diabetes", re.compile(r"\bdiabetes\b|\bsukari\b", re.I)),
    ("Asthma", re.compile(r"\basthma\b", re.I)),
    ("Urinary tract infection", re.compile(r"\buti\b|\burinary tract infection\b", re.I)),
    ("Bronchitis", re.compile(r"\bbronchitis\b", re.I)),
    ("Gastroenteritis", re.compile(r"\bgastroenteritis\b", re.I)),
    ("Typhoid", re.compile(r"\btyphoid\b", re.I)),
    ("HIV", re.compile(r"\bhiv\b", re.I)),
    ("Anaemia", re.compile(r"\b(?:anaemia|anemia)\b", re.I)),
    ("Migraine", re.compile(r"\bmigraine\b", re.I)),
    ("Sinusitis", re.compile(r"\bsinusitis\b", re.I)),
    ("Tonsillitis", re.compile(r"\btonsillitis\b", re.I)),
    ("Dyspepsia", re.compile(r"\bdyspepsia\b", re.I)),
    ("Osteoarthritis", re.compile(r"\bosteoarthritis\b|\barthritis\b", re.I)),
    ("Influenza", re.compile(r"\binfluenza\b|\bflu\b", re.I)),
    ("Ulcer", re.compile(r"\bulcers?\b", re.I)),
]

SURGERY_MAP: list[tuple[str, re.Pattern[str]]] = [
    ("Appendicectomy", re.compile(r"\bappendi(?:x|cectomy)|\bappendectomy\b", re.I)),
    ("Caesarean section", re.compile(r"\bc-?section\b|\bcaesarean\b|\bcesarean\b|\bcs\b", re.I)),
    ("Cholecystectomy", re.compile(r"\bgall\s?bladder\b|\bcholecystectomy\b", re.I)),
    ("Hernia repair", re.compile(r"\bhernia\b", re.I)),
    ("Tonsillectomy", re.compile(r"\btonsils?\b|\btonsillectomy\b", re.I)),
]

SYMPTOM_RE = _alt(
    """cough coughing fever pain ache headache vomit vomiting diarrhoea diarrhea nausea tired tiredness weak
    weakness dizzy dizziness sweating sweats breathless breathlessness swelling rash itching bleeding blood
    kikohozi nakohoa homa kutapika kuhara kuuma maumivu uchovu kuchoka kizunguzungu""".split()
    + ["not eating", "loss of appetite", "short of breath", "burning"]
)
