"""Speech evaluation: WER/CER (overall and by language).
"""
from __future__ import annotations

import asyncio
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from app.core.transcript import Transcript, parse_transcript
from app.services.speech_eval import clinical, langid
from app.services.speech_eval.align import align, edit_distance
from app.services.speech_eval.normaliser import LEVELS, NORMALISER_VERSION, Tok, normalise_transcript, normalise_words

REPORT_LEVEL = "full"
ROLE_MATCH_FLOOR = 0.2  # minimum token overlap for two turns to be considered the same turn


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def _error_class(ref: str, hyp: str, ref_class: str | None, hyp_class: str | None, lang: str) -> str:
    drugs = langid.clinical_terms()
    pair = [t for t in (ref, hyp) if t]
    if any(t[0].isdigit() for t in pair):
        return "number"
    if "negation" in (ref_class, hyp_class):
        return "negation"
    if any(t in drugs for t in pair) or (ref and hyp and any(
        SequenceMatcher(None, ref, d).ratio() >= 0.8 or SequenceMatcher(None, hyp, d).ratio() >= 0.8 for d in drugs if len(d) > 5)):
        return "drug_name"
    if "allergy" in (ref_class, hyp_class):
        return "allergy_or_reaction_word"
    if ref and hyp and SequenceMatcher(None, ref, hyp).ratio() >= 0.75:
        return "swahili_spelling_or_morphology" if lang == "sw" else "near_homophone_or_spelling"
    return "other"


def _score_tokens(ref: list[Tok], hyp: list[Tok]) -> dict[str, Any]:
    rt, ht = [t.text for t in ref], [t.text for t in hyp]
    ops = align(rt, ht)
    ref_lang, hyp_lang = [langid.tag(t) for t in rt], [langid.tag(t) for t in ht]
    ref_cls, hyp_cls = clinical.classify(rt), clinical.classify(ht)
    errors: list[dict[str, Any]] = []
    num: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for kind, i, j in ops:
        counts[kind] += 1
        if kind == "match":
            continue
        lang = ref_lang[i] if i is not None else hyp_lang[j]  # type: ignore[index]
        rc = ref_cls[i] if i is not None else None
        hc = hyp_cls[j] if j is not None else None
        if kind == "sub":
            is_clin = rc is not None or hc in ("negation", "number")
        elif kind == "del":
            is_clin = rc is not None
        else:
            is_clin = hc is not None
        num[lang] += 1
        rtok, htok = (ref[i] if i is not None else None), (hyp[j] if j is not None else None)
        errors.append({
            "ref": rtok.text if rtok else "", "hyp": htok.text if htok else "", "type": kind, "lang": lang, "clinical": is_clin,
            "clinical_class": (rc or hc) if is_clin else None,
            "error_class": _error_class(rtok.text if rtok else "", htok.text if htok else "", rc, hc, lang),
            "ref_raw": rtok.raw if rtok else "", "hyp_raw": htok.raw if htok else "",
            "time": (rtok or htok).ref,  # type: ignore[union-attr]
        })
    denom = Counter(ref_lang)
    n = len(rt)
    total_err = counts["sub"] + counts["del"] + counts["ins"]
    return {
        "ops": ops, "errors": errors, "counts": counts, "ref_lang": ref_lang, "ref_cls": ref_cls, "denom": denom, "num": num,
        "n_ref": n, "wer": (total_err / n) if n else 0.0,
        "wer_lang": {l: (num[l] / denom[l] if denom[l] else None) for l in ("en", "sw", "other")},
    }


def _sim(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def role_accuracy(ref: Transcript, hyp: Transcript) -> dict[str, Any]:
    """Monotonic text alignment of turns, then role agreement over aligned pairs.

    A reference turn with no counterpart in the hypothesis (dropped, or merged into a neighbour) counts as wrong.
    """
    rt = [[t for t, _ in normalise_words(turn.text, REPORT_LEVEL)] for turn in ref.turns]
    ht = [[t for t, _ in normalise_words(turn.text, REPORT_LEVEL)] for turn in hyp.turns]
    n, m = len(rt), len(ht)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _sim(rt[i - 1], ht[j - 1])
            best = max(dp[i - 1][j], dp[i][j - 1])
            if s >= ROLE_MATCH_FLOOR:
                best = max(best, dp[i - 1][j - 1] + s)
            dp[i][j] = best
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        s = _sim(rt[i - 1], ht[j - 1])
        if s >= ROLE_MATCH_FLOOR and abs(dp[i][j] - (dp[i - 1][j - 1] + s)) < 1e-12:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif abs(dp[i][j] - dp[i - 1][j]) < 1e-12:
            i -= 1
        else:
            j -= 1
    correct = sum(ref.turns[a].role == hyp.turns[b].role for a, b in pairs)
    return {"role_accuracy": correct / n if n else 0.0, "turns_reference": n, "turns_matched": len(pairs), "turns_role_correct": correct}


def evaluate_pair(ref: Transcript, hyp: Transcript) -> dict[str, Any]:
    ref_toks, hyp_toks = normalise_transcript(ref, REPORT_LEVEL), normalise_transcript(hyp, REPORT_LEVEL)
    s = _score_tokens(ref_toks, hyp_toks)
    ref_str, hyp_str = " ".join(t.text for t in ref_toks), " ".join(t.text for t in hyp_toks)
    cer = edit_distance(ref_str, hyp_str) / len(ref_str) if ref_str else 0.0

    clin_positions = [i for i, c in enumerate(s["ref_cls"]) if c is not None]
    clin_errors = [e for e in s["errors"] if e["clinical"]]
    cter = len(clin_errors) / len(clin_positions) if clin_positions else 0.0

    sensitivity = {}
    for level in LEVELS:
        sc = _score_tokens(normalise_transcript(ref, level), normalise_transcript(hyp, level))
        sensitivity[level] = {"wer_overall": _r(sc["wer"]), "wer_sw": _r(sc["wer_lang"]["sw"]), "wer_en": _r(sc["wer_lang"]["en"]),
                              "errors": sc["counts"]["sub"] + sc["counts"]["del"] + sc["counts"]["ins"], "reference_tokens": sc["n_ref"]}

    roles = role_accuracy(ref, hyp)
    by_class = Counter(e["error_class"] for e in s["errors"])
    clin_by_class = Counter(e["clinical_class"] for e in clin_errors)
    return {
        "normaliser_version": NORMALISER_VERSION,
        "normaliser_level": REPORT_LEVEL,
        "wer_overall": _r(s["wer"]),
        "wer_en": _r(s["wer_lang"]["en"]),
        "wer_sw": _r(s["wer_lang"]["sw"]),
        "wer_other": _r(s["wer_lang"]["other"]),
        "cer_overall": _r(cer),
        "clinical_token_error_rate": _r(cter),
        "role_accuracy": _r(roles["role_accuracy"]),
        "counts": {
            "reference_tokens": s["n_ref"], "hypothesis_tokens": len(hyp_toks),
            "substitutions": s["counts"]["sub"], "deletions": s["counts"]["del"], "insertions": s["counts"]["ins"],
            "reference_tokens_by_lang": {l: s["denom"][l] for l in ("en", "sw", "other")},
            "errors_by_lang": {l: s["num"][l] for l in ("en", "sw", "other")},
            "clinical_reference_tokens": len(clin_positions), "clinical_errors": len(clin_errors),
            "reference_characters": len(ref_str),
            "turns": {k: roles[k] for k in ("turns_reference", "turns_matched", "turns_role_correct")},
        },
        "errors": s["errors"],
        "error_class_summary": {"all_errors": dict(sorted(by_class.items())), "clinical_errors_by_token_class": dict(sorted(clin_by_class.items()))},
        "clinical_reference_tokens": [
            {"token": ref_toks[i].text, "class": s["ref_cls"][i], "time": ref_toks[i].ref} for i in clin_positions
        ],
        "normaliser_sensitivity": sensitivity,
        "language_tagging_method": langid.METHOD,
        "clinical_token_rule": clinical.RULE,
    }


async def evaluate_texts(ref_text: str, hyp_text: str, *, ref_name: str = "reference", hyp_name: str = "hypothesis") -> dict[str, Any]:
    ref = parse_transcript(ref_text, name=ref_name)
    hyp = parse_transcript(hyp_text, name=hyp_name)
    return await asyncio.to_thread(evaluate_pair, ref, hyp)
