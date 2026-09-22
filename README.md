# Eden-Care-demo: grounded clinical notes from consultations

The CLI and REST API exposes the same services.

Language and Libraries: Python, FastAPI, Pydantic v2, google-genai (Gemini). Everything is async.
---

## 1. Quick start (using pip)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # optional: add GEMINI_API_KEY to enable the Gemini engine (LLM)
./scribe check                  # exit 0 = environment ready
```

## 1. Quick start (using uv, which is what I used through out)
```bash
uv init
uv add -r requirements.txt 
cp .env.example .env
uv run python ./scribe check  
```

### The app run on the given sample consultation

```bash
uv run python ./scribe pipeline --transcript data/transcript_01.txt --register data/register.csv \
                  --source data/guideline.txt --out outputs --offline
uv run python ./scribe speech-eval --ref data/transcript_01.txt --hyp data/hyp_01.txt --out outputs/metrics_01.json
```

This creates an `outputs/` folder that holds the results: `note.json`, `resolved.json`, `knowledge.json`, `run_log.jsonl`, `metrics_01.json`.

---

## 2. Verification status (read this first)

| Area | Status |
|---|---|
| Rules engine, validator, resolver, knowledge, speech-eval, pipeline, CLI, API | Implemented and covered by tests; the CLI is exercised end to end via subprocess. |
| Gemini engine (`google-genai`) | Code complete. Tested with an **injected fake client** using the real SDK config types (`GenerateContentConfig`, `HttpOptions`), covering grounding, retries, malformed output, fallback and failure. **It has not been run against the live Gemini API**: no key or network was available where this was built. Treat the first real run as the first integration test. |
| Committed outputs | Produced by the **rules engine** (`--offline`), not by Gemini. |
| Default model | `gemini-3.5-flash`, configurable via `GEMINI_MODEL`. When I checked, `gemini-2.5-flash` was listed for shutdown in October 2026, so I did not default to it. Verify the current model list before relying on this. |

**Privacy note.** With a key set, `extract` sends transcript *text* to Google's API. That is fine for synthetic data. For real patient consultations, use `--engine rules`, or replace the client with an on-premises model, before any clinical use; see Part E for the data-residency constraint.

---

## 3. CLI

| Command | What it does | Exit codes (sample) |
|---|---|---|
| `uv run python ./scribe check` | Python version, dependencies, bundled files, a rules-engine self-test; reports whether Gemini is configured | 0 ready, 1 missing something |
| `uv run python ./scribe extract --transcript T --out note.json [--engine auto\|gemini\|rules] [--offline]` | transcript -> grounded note | 0 ok, 2 bad input, 3 extraction failure |
| `uv run python ./scribe validate --transcript T --note note.json` | prove the note is grounded; prints each violation | 0 valid, 1 rejected, 2 unreadable/malformed input |
| `uv run python ./scribe resolve --note note.json --register register.csv --out resolved.json` | deterministic coding, no model | 0 ok, 2 bad register/note |
| `uv run python ./scribe knowledge --source guideline.txt --out knowledge.json` | cited rows from a guideline | 0 ok, 2 bad source |
| `uv run python ./scribe speech-eval --ref R --hyp H --out metrics.json` | speech metrics | 0 ok, 2 bad input |
| `uv run python ./scribe pipeline --transcript T --register R --source S --out DIR [--offline]` | all stages + `run_log.jsonl` | 0 ok; non-zero naming the failing stage and input |
| `uv run python./scribe serve` or `uv run uvicorn app.api.main:app` | run the REST API | |

**Engines (check example env file).** `auto` (default) uses Gemini when a key is present and otherwise the rules engine. `--engine gemini` fails non-zero if Gemini is unavailable or its output cannot be grounded. `--offline` (= `--engine rules`) is deterministic and needs no network. There is no special-casing of `transcript_01.txt`.

---

### The five rules and how each is enforced

1. **Nothing recorded that the patient did not say.** Every element carries `span.ref` + `span.text`, a verbatim substring of exactly one transcript line. The validator rejects any element whose span is not verbatim.
2. **No code from a model.** Extractors never emit codes. The contract's three code patterns (ICD, ATC, Eden Care) are checked by the validator over every key and string of the note, and again per element in the Gemini grounding gate. Elements containing a code-like string are dropped.
3. **No model in the resolver.** `app/services/resolver/` is pure standard-library code: exact/phrase/token-bag/edit-distance matching against the register.
4. **No validator that lets a changed number through.** Every number in `value` (or `entity`) must appear in the span, as digits *or* spoken words, in either direction (`3 weeks` - `three weeks`, `2019` - `twenty nineteen`, `128/82` - `128 over 82`). Swahili numerals 1–10 are understood too. A changed number is rejected even if written as a word.
5. **No reported metric the committed tool cannot reproduce.** `metrics_01.json` is regenerated and compared for equality.
