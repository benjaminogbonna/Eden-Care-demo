"""Deterministic, rules based extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.numbers import digitize, spoken_to_digits
from app.core.patterns import (
    ACK_RE,
    FAMILY_RE,
    FAMILY_TOPIC_REL_RE,
    NEGATION_ONLY_RE,
    find_code_like,
    is_thinking_aloud,
)
from app.core.text import Frag, split_regex, split_sentences
from app.core.transcript import CLINICIAN_ROLES, Transcript, Turn
from app.domain.note import KIND_REJECTED, NOT_STATED, SECTIONS, Element, Span
from app.services.extraction import lexicon as lx

GREETING_RE = re.compile(r"^\s*(?:habari|hujambo|shikamoo|hello|hi|good (?:morning|afternoon)|karibu)\b[^.?!]{0,25}[.!]?\s*$", re.I)

_TOPICS: list[tuple[str, re.Pattern[str]]] = [
    ("allergies", re.compile(r"allerg|mzio", re.I)),
    ("surgery", re.compile(r"surger|operation|operated|upasuaji", re.I)),
    ("medications", re.compile(r"medicin|medication|tablet|\bdrugs?\b|\bdawa\b|taking any|on any (?:treatment|medic)|prescri", re.I)),
    ("family", re.compile(r"family|familia|ukoo|relatives|runs in", re.I)),
    ("social", re.compile(r"smok|drink|alcohol|sigara|pombe|tobacco|occupation|\bmiraa\b|\bkhat\b", re.I)),
    ("pmh", re.compile(r"medical (?:condition|problem|history)|past (?:illness|medical)|chronic|diagnosed with|hospitali[sz]ed|admitted|magonjwa|other (?:conditions|illness)", re.I)),
    ("chief", re.compile(r"what brings you|what is troubling|what seems to be|how can i help|what is the (?:problem|complaint)|tatizo|kinakusumbua|unaumwa|what.s wrong|what is bothering", re.I)),
    ("identity", re.compile(r"who (?:have you come|came|is with|has come)|nani amekuja|come with", re.I)),
    ("hpi", re.compile(r"worse|better|relie|aggravat|radiat|how long|since when|when did|how often|how bad|constant|come and go|severity|scale|trigger", re.I)),
    ("ros", re.compile(r"^\s*(?:okay[,.\s]+)?(?:any|do (?:you|he|she) have|have you (?:had|noticed|been)|has (?:he|she) had|is there|are there|unapata)\b", re.I)),
]


def detect_topic(question: str) -> str:
    for name, rx in _TOPICS:
        if rx.search(question):
            return name
    return "unknown"


def ros_items(question: str) -> list[str]:
    s = question.strip().rstrip("?")
    s = re.sub(r"^(?:okay|ok|and|so|now)[,\s]+", "", s, flags=re.I)
    s = re.sub(
        r"^(?:any|do you have any|do you have|have you had any|have you had|have you noticed any|have you noticed|"
        r"have you been having|is there any|are there any|unapata|has she had any|has he had any|"
        r"does (?:he|she) have any)\s+",
        "", s, flags=re.I,
    )
    return [i.strip().lower() for i in re.split(r",|\bor\b|\band\b|\bna\b|\bau\b", s, flags=re.I) if i.strip()]


# clinician patterns
_LEADIN = r"(?:(?:and|so|then)\s+)?(?:(?:we|i|you)(?:'ll|\s+will|\s+should|\s+shall|\s+need to|\s+must)\s+|let'?s\s+|please\s+)?"
_PLAN_VERBS = (
    r"start|stop|prescribe|give|refer|check|order|send|continue|advise|admit|repeat|arrange|book|schedule|"
    r"come back|return|follow[- ]?up|review|avoid|increase|reduce|switch|change|take"
)
PLAN_START_RE = re.compile(rf"^\s*{_LEADIN}(?:{_PLAN_VERBS})\b", re.I)
LEADIN_RE = re.compile(rf"^\s*{_LEADIN}", re.I)

_DIFF = re.compile(r"\b(?:could be|might be|may be|possibly|possible|differential|rule out|cannot exclude|can't exclude|not (?:yet )?sure|versus|vs)\b", re.I)
_PROB = re.compile(r"\b(?:most likely|probably|likely|i think|we think|i suspect|suspect(?:ed)?|seems|looks like|appears|consistent with|suggests?|suggestive of|in keeping with)\b", re.I)
_CONF = re.compile(r"\b(?:diagnos(?:is|ed)|confirmed|you have|you've got|you are having|this is|it is|it's|positive for)\b", re.I)
_STRONG = re.compile(r"\b(?:diagnos(?:is|ed)|most likely|probably|i think (?:this|it)|i suspect|consistent with)\b", re.I)
_TAIL = re.compile(r"\s+((?:related to|due to|secondary to|caused by|associated with)\b[^,.;]*)", re.I)
_EXAM = re.compile(
    r"tender|\bsoft\b|guarding|rebound|distended|crackles|crepitations|wheez|murmur|\bclear\b|swollen|swelling|"
    r"enlarged|pallor|\bpale\b|jaundice|oedema|edema|reflex|bowel sounds|abnormal|\bnormal\b|rigid|\bmass\b|\blump\b|"
    r"erythema|inflamed|air entry|breath sounds|dehydrat",
    re.I,
)

_VITALS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:bp|blood pressure)\b\D{0,15}?(\d{2,3})\s*(?:/|over)\s*(\d{2,3})"), "BP {1}/{2}"),
    (re.compile(r"\b(?:pulse|heart rate|hr)\b\D{0,12}?(\d{2,3})"), "Pulse {1}"),
    (re.compile(r"\btemp(?:erature)?\b\D{0,12}?(\d{2}(?:\.\d)?)"), "Temperature {1}"),
    (re.compile(r"\b(?:respiratory rate|respiration rate|resp rate|rr)\b\D{0,12}?(\d{1,2})"), "Respiratory rate {1}"),
    (re.compile(r"\b(?:oxygen saturation|saturation|sats?|spo2)\b\D{0,15}?(\d{2,3})(\s*(?:%|percent))?"), "Oxygen saturation {1}{pct}"),
    (re.compile(r"\bweight\b\D{0,10}?(\d{2,3}(?:\.\d)?)(\s*(?:kg|kilos?|kilograms?))?"), "Weight {1}{kg}"),
]


def _vital_value(fragment: str) -> str | None:
    low = spoken_to_digits(fragment.lower())
    for rx, fmt in _VITALS:
        m = rx.search(low)
        if not m:
            continue
        groups = m.groups()
        out = fmt.replace("{1}", groups[0]).replace("{2}", groups[1] if len(groups) > 1 and "{2}" in fmt else "")
        out = out.replace("{pct}", "%" if fmt.endswith("{pct}") and len(groups) > 1 and groups[1] else "")
        out = out.replace("{kg}", " kg" if fmt.endswith("{kg}") and len(groups) > 1 and groups[1] else "")
        return out
    return None


def _terms(rx_list, text: str) -> list[tuple[str, int, int]]:
    hits: list[tuple[str, int, int]] = []
    for display, rx in rx_list:
        for m in rx.finditer(text):
            hits.append((display, m.start(), m.end()))
    hits.sort(key=lambda h: (h[1], -(h[2] - h[1])))
    kept: list[tuple[str, int, int]] = []
    for h in hits:
        if kept and h[1] < kept[-1][2]:
            continue
        if any(k[0] == h[0] for k in kept):
            continue
        kept.append(h)
    return kept


def dx_terms(text: str) -> list[tuple[str, int, int]]:
    return _terms(lx.DIAGNOSES, text)


_PMH_EXTRA = [("Hypertension", re.compile(r'(?<!blood )\bpressure\b|\bhypertension\b', re.I))]


def _certainty(text: str) -> str | None:
    if _DIFF.search(text):
        return "differential"
    if _PROB.search(text):
        return "probable"
    if _CONF.search(text):
        return "confirmed"
    return None


def _clean_span(text: str) -> str:
    return text.strip().rstrip(" .!?")


@dataclass
class _State:
    topic: str | None = None
    question: str = ""
    items: list[str] = field(default_factory=list)
    chief_done: bool = False


class _Builder:
    def __init__(self) -> None:
        self.items: dict[str, list[Element]] = {s: [] for s in SECTIONS}
        self.warnings: list[str] = []
        self._seen: set[tuple] = set()

    def add(self, section: str, value: str, turn: Turn, span_text: str, *, confidence: float = 1.0,
            certainty: str | None = None, kind: str | None = None, entity: str | None = None) -> None:
        value, span_text = value.strip().rstrip(" ."), _clean_span(span_text)
        if not value or not span_text:
            return
        for label, txt in (("value", value), ("span", span_text), ("entity", entity or "")):
            if find_code_like(txt):
                self.warnings.append(f"{turn.ref}: dropped {section} element because its {label} looks like a code")
                return
        key = (section, value, turn.ref, span_text)
        if key in self._seen:
            return
        self._seen.add(key)
        self.items[section].append(
            Element(value=value, span=Span(ref=turn.ref, text=span_text), confidence=confidence,
                    attribution=turn.role.lower(), certainty=certainty, kind=kind, entity=entity)
        )

    def note(self) -> dict:
        return {s: ([e.to_dict() for e in els] if els else NOT_STATED) for s, els in self.items.items()}


def _segments_by_match(text: str, matches: list[re.Match]) -> list[str]:
    """Cut `text` between consecutive matches at the last delimiter (comma, 'and', 'na') so each piece holds one match."""
    if len(matches) < 2:
        return [text]
    cuts: list[tuple[int, int]] = []
    for prev, nxt in zip(matches, matches[1:]):
        gap = text[prev.end():nxt.start()]
        found = list(re.finditer(r",|;|\band\b|\bna\b", gap, flags=re.I))
        if found:
            d = found[-1]
            cuts.append((prev.end() + d.start(), prev.end() + d.end()))
    if not cuts:
        return [text]
    pieces, pos = [], 0
    for a, b in cuts:
        pieces.append(text[pos:a])
        pos = b
    pieces.append(text[pos:])
    return [p.strip(" ,;") for p in pieces if p.strip(" ,;")]


class RulesExtractor:
    engine = "rules"

    def extract(self, transcript: Transcript) -> tuple[dict, list[str]]:
        b, st = _Builder(), _State()
        for turn in transcript.turns:
            if turn.role in CLINICIAN_ROLES:
                self._clinician(turn, st, b)
            else:
                self._history(turn, st, b)
        return b.note(), b.warnings

    # clinician turns
    def _clinician(self, turn: Turn, st: _State, b: _Builder) -> None:
        sentences = split_sentences(turn.text)
        questions = [s.text for s in sentences if s.text.endswith("?")]
        if questions:
            st.question = " ".join(questions)
            st.topic = detect_topic(st.question)
            st.items = ros_items(st.question) if st.topic == "ros" else []
        else:
            st.topic, st.question, st.items = None, "", []
        for s in sentences:
            if not s.text.endswith("?"):
                self._statement(turn, s, b)

    def _statement(self, turn: Turn, s: Frag, b: _Builder) -> None:
        text = s.text
        if GREETING_RE.match(text):
            return
        if is_thinking_aloud(text):
            self._thinking(turn, text, b)
            return
        segs = split_regex(text, r",|;|\s+so\s+|\s+then\s+|\s+and\s+", base=s.start)
        plan_idx = [i for i, g in enumerate(segs) if PLAN_START_RE.match(g.text)]
        if not plan_idx:
            if not self._assessment(turn, text, b):
                self._fragments(turn, text, segs, b)
            return
        first = plan_idx[0]
        if first > 0:
            pre = turn.text[segs[0].start:segs[first - 1].end]
            if not self._assessment(turn, pre, b):
                self._fragments(turn, pre, segs[:first], b)
        for n, i in enumerate(plan_idx):
            end_seg = (plan_idx[n + 1] - 1) if n + 1 < len(plan_idx) else len(segs) - 1
            self._plan(turn, turn.text[segs[i].start:segs[end_seg].end], b)

    def _thinking(self, turn: Turn, sentence: str, b: _Builder) -> None:
        for display, _, _ in dx_terms(sentence):
            b.add("assessment", f"{display} considered by clinician, not pursued", turn, sentence,
                  confidence=0.8, certainty="differential", kind=KIND_REJECTED, entity=display)
        for rx in (lx.DRUG_RE, lx.LAB_RE):
            for m in rx.finditer(sentence):
                b.add("plan", f"{m.group().lower()} considered by clinician, not pursued", turn, sentence,
                      confidence=0.8, kind=KIND_REJECTED, entity=m.group().lower())

    def _assessment(self, turn: Turn, unit: str, b: _Builder) -> bool:
        cert = _certainty(unit)
        if cert is None:
            return False
        terms = dx_terms(unit)
        if not terms and not _STRONG.search(unit):
            return False
        if FAMILY_RE.search(unit):
            b.warnings.append(f"{turn.ref}: assessment sentence mentions a family member; not recorded")
            return True
        if terms:
            for display, _, end in terms:
                tail = _TAIL.search(unit[end:]) if len(terms) == 1 else None
                value = display + (f" {tail.group(1).strip()}" if tail else "")
                b.add("assessment", value, turn, unit, confidence=0.95, certainty=cert, entity=display)
        else:
            body = re.split(_PROB.pattern + "|" + _CONF.pattern, unit, flags=re.I)[-1] if False else unit
            b.add("assessment", lx.capitalise(body), turn, unit, confidence=0.7, certainty=cert)
        return True

    def _fragments(self, turn: Turn, unit: str, segs: list[Frag], b: _Builder) -> None:
        exam_sentence = bool(_EXAM.search(unit))
        for g in segs:
            vital = _vital_value(g.text)
            if vital:
                b.add("vitals", vital, turn, g.text, confidence=1.0)
            elif exam_sentence and len(g.text) > 3 and _EXAM.search(unit):
                b.add("examination", lx.capitalise(g.text), turn, g.text, confidence=1.0)

    def _plan(self, turn: Turn, span_text: str, b: _Builder) -> None:
        body = LEADIN_RE.sub("", span_text, count=1)
        value = lx.capitalise(digitize(body)).rstrip(" .")
        entity = None
        for rx in (lx.DRUG_RE, lx.LAB_RE):
            m = rx.search(body)
            if m:
                entity = m.group().lower()
                break
        b.add("plan", value, turn, span_text, confidence=1.0, entity=entity)

    # patient / companion turns
    def _history(self, turn: Turn, st: _State, b: _Builder) -> None:
        sentences = split_sentences(turn.text)
        if st.topic == "allergies":
            self._allergies(turn, turn.text, b)
            return
        for s in sentences:
            text = s.text
            neg = bool(NEGATION_ONLY_RE.match(text))
            if not neg and (ACK_RE.match(text) or GREETING_RE.match(text)):
                continue
            t = st.topic
            if t == "chief":
                if neg:
                    continue
                if not st.chief_done:
                    st.chief_done = True
                    value = lx.gloss(text)
                    b.add("chief_complaint", value, turn, text, confidence=0.9)
                    b.add("history_of_presenting_illness", value, turn, text, confidence=0.9)
                else:
                    self._content(turn, text, b, st)
            elif t == "hpi":
                self._hpi(turn, text, neg, st, b)
            elif t == "ros":
                self._ros(turn, text, neg, st, b)
            elif t == "medications":
                self._meds(turn, text, neg, b)
            elif t == "surgery":
                self._surgery(turn, text, neg, b)
            elif t == "family":
                self._family(turn, text, neg, st, b)
            elif t == "social":
                self._social(turn, text, neg, st, b)
            elif t == "pmh":
                self._pmh(turn, text, neg, b)
            elif neg:
                continue
            else:
                self._content(turn, text, b, st, allow_family=(t != "identity"))

    def _content(self, turn: Turn, text: str, b: _Builder, st: _State, allow_family: bool = True) -> None:
        """Topic-less classification. Unclassifiable sentences are dropped."""
        if re.search(r"allerg", text, re.I):
            self._allergies(turn, text, b)
        elif allow_family and FAMILY_RE.search(text) and (
            dx_terms(text) or lx.SYMPTOM_RE.search(text) or re.search(r"\b(?:had|has|alikuwa|died|passed)\b", text, re.I)
        ):
            b.add("family_history", lx.gloss(text), turn, text, confidence=0.9)
        elif re.search(r"\b(?:i take|i am taking|i'm taking|nakunywa|natumia|i use)\b", text, re.I) or lx.DRUG_RE.search(text):
            self._meds(turn, text, False, b)
        elif lx.SYMPTOM_RE.search(text):
            b.add("history_of_presenting_illness", lx.gloss(text), turn, text, confidence=0.9)

    def _hpi(self, turn: Turn, text: str, neg: bool, st: _State, b: _Builder) -> None:
        q = st.question.lower()
        if neg:
            if not re.search(r"\d", q):
                b.add("history_of_presenting_illness", f"Answered no to: {st.question.rstrip('?')}", turn, text, confidence=0.8)
            return
        g = lx.gloss(text)
        short = len(text.split()) <= 6
        if short and re.search(r"worse", q):
            g = "Worse " + g[:1].lower() + g[1:]
        elif short and re.search(r"better|relie", q):
            g = "Better " + g[:1].lower() + g[1:]
        b.add("history_of_presenting_illness", g, turn, text, confidence=0.9)

    def _ros(self, turn: Turn, text: str, neg: bool, st: _State, b: _Builder) -> None:
        items = st.items
        if neg:
            for it in items:
                b.add("review_of_systems", f"No {it}", turn, text, confidence=0.9)
            return
        words = set(re.findall(r"[a-z]+", text.lower()))
        generic = {"loss", "the", "in", "of", "a", "any", "and", "or", "is", "are", "have", "had", "ni", "any"}
        best, score = None, 0
        for it in items:
            overlap = len((words & set(re.findall(r"[a-z]+", it))) - generic)
            if overlap > score:
                best, score = it, overlap
        explicit = re.match(r"^\s*(?:no|hapana|sina|hakuna)\b", text, re.I)
        normal = re.search(r"\b(?:normal|fine|sawa|ok|okay|poa|good|nzuri)\b", text, re.I)
        if best and explicit:
            b.add("review_of_systems", f"No {best}", turn, text, confidence=1.0)
        elif best and normal:
            g = lx.gloss(text)
            b.add("review_of_systems", f"No {best} (patient: {g[:1].lower() + g[1:]})", turn, text, confidence=0.85)
        else:
            self._content(turn, text, b, st)

    def _meds(self, turn: Turn, text: str, neg: bool, b: _Builder) -> None:
        if neg:
            b.add("medication_history", "No current medications reported", turn, text, confidence=0.9)
            return
        for seg in _segments_by_match(text, list(lx.DRUG_RE.finditer(text))):
            m = lx.DRUG_RE.search(seg)
            b.add("medication_history", lx.gloss(seg), turn, seg, confidence=0.95 if m else 0.85,
                  entity=m.group().lower() if m else None)

    def _allergies(self, turn: Turn, text: str, b: _Builder) -> None:
        text = text.strip()
        first = split_sentences(text)[0].text if text else ""
        if NEGATION_ONLY_RE.match(first) and len(split_sentences(text)) == 1:
            b.add("allergies", "No known allergies", turn, text, confidence=0.9)
            return
        allergens: list[str] = []
        for m in lx.ALLERGEN_RE.finditer(text):
            if m.group().lower() not in [a.lower() for a in allergens]:
                allergens.append(m.group())
        reaction = lx.REACTION_RE.search(text)
        timing = re.search(r"\b(?:last (?:year|month|week)|\w+ (?:years?|months?) ago|as a child|when i was (?:a )?child)\b", text, re.I)
        if not allergens:
            b.add("allergies", lx.gloss(text), turn, text, confidence=0.8)
            return
        for a in allergens:
            value = lx.capitalise(a)
            if reaction:
                value += f" — {reaction.group().lower()}"
            if timing:
                value += f" ({timing.group().lower()})"
            b.add("allergies", value, turn, text, confidence=0.95, entity=a.lower())

    def _surgery(self, turn: Turn, text: str, neg: bool, b: _Builder) -> None:
        if neg:
            b.add("past_surgical_history", "No previous surgery reported", turn, text, confidence=0.9)
            return
        hits = [(name, m) for name, rx in lx.SURGERY_MAP for m in rx.finditer(text)]
        if not hits:
            b.add("past_surgical_history", lx.gloss(text), turn, text, confidence=0.8)
            return
        for name, rx in lx.SURGERY_MAP:
            pieces = _segments_by_match(text, list(rx.finditer(text))) if len(list(rx.finditer(text))) > 1 else [text]
            for piece in pieces:
                m = rx.search(piece)
                if not m:
                    continue
                said = m.group().lower()
                rest = lx.gloss((piece[:m.start()] + piece[m.end():]).strip(" ,;"))
                head = name if said == name.lower() else f"{name} ({said})"
                b.add("past_surgical_history", f"{head}, {rest}" if rest else head, turn, piece, confidence=0.9, entity=name.lower())

    def _family(self, turn: Turn, text: str, neg: bool, st: _State, b: _Builder) -> None:
        if neg:
            m = re.search(r"\bwith (.+?)\?", st.question, re.I)
            value = f"No family history of {m.group(1)}" if m and not re.search(r"\d", m.group(1)) else "No family history reported"
            b.add("family_history", value, turn, text, confidence=0.9)
            return
        g = lx.gloss(text)
        b.add("family_history", g if FAMILY_TOPIC_REL_RE.search(text) else f"Family history: {g}", turn, text, confidence=0.9)

    def _social(self, turn: Turn, text: str, neg: bool, st: _State, b: _Builder) -> None:
        if neg:
            q = st.question.lower()
            parts = [w for w, k in (("smoking", "smok"), ("alcohol", "drink|alcohol|pombe")) if re.search(k, q)]
            value = "Denies " + " and ".join(parts) if parts else "Denies (as asked)"
            b.add("social_history", value, turn, text, confidence=0.85)
            return
        b.add("social_history", lx.gloss(text), turn, text, confidence=0.9)

    def _pmh(self, turn: Turn, text: str, neg: bool, b: _Builder) -> None:
        if neg:
            b.add("past_medical_history", "No past medical conditions reported", turn, text, confidence=0.85)
            return
        terms = _terms(lx.DIAGNOSES + _PMH_EXTRA, text)
        if not terms:
            b.add("past_medical_history", lx.gloss(text), turn, text, confidence=0.8)
            return
        for display, _, _ in terms:
            b.add("past_medical_history", display, turn, text, confidence=0.9, entity=display)
