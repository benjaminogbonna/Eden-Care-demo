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

## 2. CLI

| Command | What it does | Exit codes (sample) |
|---|---|---|
| `uv run python ./scribe check` | Python version, dependencies, bundled files, a rules-engine self-test; reports whether Gemini is configured | 0 ready, 1 missing something |
| `uv run python ./scribe extract --transcript ./data/transcript_01.txt --out note.json [--engine auto\|gemini\|rules] [--offline]` | transcript -> grounded note | 0 ok, 2 bad input, 3 extraction failure |
| `uv run python ./scribe validate --transcript ./data/transcript_01.txt  --note ./outputs/note.json` | prove the note is grounded; prints each violation | 0 valid, 1 rejected, 2 unreadable/malformed input |
| `uv run python ./scribe resolve --note note.json --register register.csv --out resolved.json` | deterministic coding, no model | 0 ok, 2 bad register/note |
| `uv run python ./scribe knowledge --source guideline.txt --out knowledge.json` | cited rows from a guideline | 0 ok, 2 bad source |
| `uv run python ./scribe speech-eval --ref R --hyp H --out metrics.json` | speech metrics | 0 ok, 2 bad input |
| `uv run python ./scribe pipeline --transcript ./data/transcript_01.txt --register R --source S --out DIR [--offline]` | all stages + `run_log.jsonl` | 0 ok; non-zero naming the failing stage and input |
| `uv run python./scribe serve` or `uv run uvicorn app.api.main:app` | run the REST API | |

**Engines (check example env file).** `auto` (default) uses Gemini when a key is present and otherwise the rules engine. `--engine gemini` fails non-zero if Gemini is unavailable or its output cannot be grounded. `--offline` (= `--engine rules`) is deterministic and needs no network. There is no special-casing of `transcript_01.txt`.

---

### The five rules and how each is enforced

1. **Nothing recorded that the patient did not say.** Every element carries `span.ref` + `span.text`, a verbatim substring of exactly one transcript line. The validator rejects any element whose span is not verbatim.
2. **No code from a model.** Extractors never emit codes. The contract's three code patterns (ICD, ATC, Eden Care) are checked by the validator over every key and string of the note, and again per element in the Gemini grounding gate. Elements containing a code-like string are dropped.
3. **No model in the resolver.** `app/services/resolver/` is pure standard-library code: exact/phrase/token-bag/edit-distance matching against the register.
4. **No validator that lets a changed number through.** Every number in `value` (or `entity`) must appear in the span, as digits *or* spoken words, in either direction (`3 weeks` - `three weeks`, `2019` - `twenty nineteen`, `128/82` - `128 over 82`). Swahili numerals 1–10 are understood too. A changed number is rejected even if written as a word.
5. **No reported metric the committed tool cannot reproduce.** `metrics_01.json` is regenerated and compared for equality.

### Known limits of the rules engine

The rules engine is topic-driven: it depends on the doctor asking recognisable questions (allergies, medicines, surgeries, family, etc) and on bilingual lexicons.

---

## 3. REST API (using uv)

Run with `uv run python ./scribe serve` (or `uv run uvicorn app.api.main:app`). Interactive docs are generated automatically at **`/docs`** (Swagger) and **`/redoc`**; and the schema is at `/openapi.json`.

Every response, success or error, has the same format:

```json
{ "success": true, "message": "note extracted (engine=rules)", "data": { ... }, "errors": [] }
{ "success": false, "message": "note rejected with 1 violation(s)", "data": null,
  "errors": [ { "code": "number_not_in_span", "message": "...", "field": "$.vitals[1]" } ] }
```

| Endpoint | Purpose | Failure status |
|---|---|---|
| `GET /health` | liveness; reports whether API is running and Gemini is configured | |
| `POST /api/v1/extract` | `{transcript, engine?}` -> note (+ engine, model, prompt hash, fallback reason, warnings) | 400 bad transcript, 502 Gemini failure |
| `POST /api/v1/validate` | `{transcript, note}` -> `{valid: true}` or violations in `errors` | 422 rejected |
| `POST /api/v1/resolve` | `{note, register_csv?}` -> resolved note | 400 bad register/note |
| `POST /api/v1/knowledge` | `{source?}` -> cited rows | 400 |
| `POST /api/v1/speech-eval` | `{reference, hypothesis}` -> metrics | 400 |
| `POST /api/v1/pipeline` | all stages; returns outputs + run log | 422 with `data.run_log` and the failing stage |

```bash
curl -s localhost:8000/api/v1/extract -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json;print(json.dumps({"transcript":open("data/transcript_01.txt").read(),"engine":"rules"}))')"
```

Security: optional API-key auth (`X-API-Key`, constant-time compare) when `API_KEY` is set (`/health` stays open); request-size cap (checked against `Content-Length`; put a reverse proxy in front for chunked uploads); CORS off unless configured; transcript text is never logged (only hashes and counts); 500s return a generic message.

### Environment configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` (alias `GOOGLE_API_KEY`) | unset | Enables the Gemini engine |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model name |
| `GEMINI_TIMEOUT_SECONDS` / `GEMINI_MAX_RETRIES` | `90` / `3` | Retries (exponential back-off) apply to 408/429/5xx only |
| `ENGINE` | `auto` | `auto`, `gemini` or `rules` |
| `API_KEY` | unset | Require `X-API-Key` on `/api/v1/*` |
| `CORS_ORIGINS` | empty | Comma-separated origins |
| `MAX_BODY_BYTES` | `2000000` | Request and transcript size cap |
| `HOST` / `PORT` / `LOG_LEVEL` | `127.0.0.1` / `8000` / `INFO` | Server and logging |

---

## 4. The Gemini engine

The system prompt (`prompts/extract_system.md`) states the rules above, forbids codes, and defines `entity` (a plain-English name for the resolver to match later). Structured output uses `response_schema` with `temperature=0` and `seed=0`. The model's JSON is then treated as **untrusted input**:

* `attribution` is overwritten from the real speaker of the cited line;
* any element with a non-verbatim span, a changed number, a code, a family-history leak, an invalid certainty or a wrong speaker for its section is **dropped and reported** (the same `check_element` the validator uses);
* thinking-aloud is forced to `considered_and_rejected`;
* the assembled note must then pass the full validator, otherwise the engine fails (and `auto` falls back).

The prompt hash and model are written to `run_log.jsonl`.

---

## 5. Speech evaluation

`uv run python ./scribe speech-eval` reports overall and per-language WER, CER, a **clinical-token error rate (CTER)**, turn-level role accuracy, and an error table (`ref`, `hyp`, `type`, `lang`, `clinical`, plus raw forms, timestamp and an error class).

### Result on `transcript_01.txt` vs `hyp_01.txt` (normaliser `scribe-norm/1.0`)

### Language tagging

Each reference token is tagged `en`, `sw` or `other` by (1) digits, single letters and drug/lab names -> `other`; (2) a bundled Swahili wordlist -> `sw`; (3) Swahili verb morphology (subject prefix + tense marker + stem, >= 6 letters, with an English guard list) -> `sw`; (4) everything else -> `en`.

### Clinical tokens

A reference token is clinical if it is a **negation**, **drug/lab name**, **allergy/reaction word**, **number** (dose, vital, year, duration), **dose unit or frequency** (`mg`, `once`, `daily`, `mara <numeral>`), **vital-sign name**, or **date/duration word**. 

---