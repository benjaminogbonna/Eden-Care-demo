# Tests for CLI pipeline.
import json
import os
import subprocess
import sys

import pytest

from tests.conftest import DATA, ROOT

SCRIBE = str(ROOT / "scribe")


def scribe(*args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env["SCRIBE_NO_REEXEC"] = "1"
    env.update(env_extra or {})
    return subprocess.run([sys.executable, SCRIBE, *args], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)


def test_check():
    r = scribe("check")
    assert r.returncode == 0 and "environment ready" in r.stdout


def test_extract_validate_roundtrip(tmp_path):
    note = tmp_path / "note.json"
    r = scribe("extract", "--transcript", str(DATA / "transcript_01.txt"), "--out", str(note), "--offline")
    assert r.returncode == 0 and note.exists()
    assert scribe("validate", "--transcript", str(DATA / "transcript_01.txt"), "--note", str(note)).returncode == 0


def test_extract_without_key_says_so_loudly(tmp_path):
    r = scribe("extract", "--transcript", str(DATA / "transcript_01.txt"), "--out", str(tmp_path / "n.json"))
    assert r.returncode == 0 and "GEMINI_API_KEY not set" in r.stderr and "engine=rules" in r.stderr


def test_engine_gemini_without_key_exits_non_zero(tmp_path):
    r = scribe("extract", "--transcript", str(DATA / "transcript_01.txt"), "--out", str(tmp_path / "n.json"), "--engine", "gemini")
    assert r.returncode == 3 and "GEMINI_API_KEY" in r.stderr and not (tmp_path / "n.json").exists()


def test_validate_rejects_tampered_and_malformed(tmp_path, note1):
    note1["vitals"][1]["value"] = "Pulse 67"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(note1))
    r = scribe("validate", "--transcript", str(DATA / "transcript_01.txt"), "--note", str(p))
    assert r.returncode == 1 and "number_not_in_span" in r.stderr and "REJECTED" in r.stderr
    p.write_text("{not json")
    assert scribe("validate", "--transcript", str(DATA / "transcript_01.txt"), "--note", str(p)).returncode == 2
    assert scribe("validate", "--transcript", str(DATA / "transcript_01.txt"), "--note", str(tmp_path / "missing.json")).returncode == 2


def test_unreadable_transcript_is_a_clean_error(tmp_path):
    bad = tmp_path / "t.txt"
    bad.write_text("this is not a transcript")
    r = scribe("extract", "--transcript", str(bad), "--out", str(tmp_path / "n.json"), "--offline")
    assert r.returncode == 2 and "expected '[mm:ss] SPEAKER: text'" in r.stderr and "Traceback" not in r.stderr


def test_knowledge_and_speech_eval(tmp_path):
    assert scribe("knowledge", "--source", str(DATA / "guideline.txt"), "--out", str(tmp_path / "k.json")).returncode == 0
    r = scribe("speech-eval", "--ref", str(DATA / "transcript_01.txt"), "--hyp", str(DATA / "hyp_01.txt"), "--out", str(tmp_path / "m.json"))
    assert r.returncode == 0 and "clinical-token error rate" in r.stdout
    assert json.loads((tmp_path / "m.json").read_text()) == json.loads((ROOT / "outputs" / "metrics_01.json").read_text())


def read_log(out):
    return [json.loads(l) for l in (out / "run_log.jsonl").read_text().splitlines()]


def test_pipeline_success_and_log(tmp_path):
    r = scribe("pipeline", "--transcript", str(DATA / "transcript_01.txt"), "--register", str(DATA / "register.csv"),
               "--source", str(DATA / "guideline.txt"), "--out", str(tmp_path), "--offline")
    assert r.returncode == 0, r.stderr
    log = read_log(tmp_path)
    assert [e["stage"] for e in log] == ["transcribe", "extract", "validate", "resolve", "knowledge"]
    assert len({e["run_id"] for e in log}) == 1
    assert {e["stage"]: e["status"] for e in log} == {"transcribe": "skipped", "extract": "ok", "validate": "ok", "resolve": "ok", "knowledge": "ok"}
    assert all(e["input_hash"] and e["output_hash"] for e in log if e["status"] == "ok")
    for name in ("note.json", "resolved.json", "knowledge.json"):
        assert (tmp_path / name).exists()


def test_pipeline_on_companion_transcript(tmp_path):
    r = scribe("pipeline", "--transcript", str(DATA / "transcript_companion_test.txt"), "--register", str(DATA / "register.csv"),
               "--source", str(DATA / "guideline.txt"), "--out", str(tmp_path), "--offline")
    assert r.returncode == 0, r.stderr


BROKEN = {
    "kind_conflict": "kind,code,name,synonyms\ndrug,A1,Alpha,x\nlab,A1,Beta,y\n",
    "bad_row": "kind,code,name,synonyms\ndrug,A1,Alpha\n",
    "no_header": "just,some,text\n",
    "empty": "",
}


@pytest.mark.parametrize("case", BROKEN)
def test_broken_register_fails_loudly_naming_stage_and_file(tmp_path, case):
    reg = tmp_path / "register.csv"
    reg.write_text(BROKEN[case])
    out = tmp_path / "out"
    r = scribe("pipeline", "--transcript", str(DATA / "transcript_01.txt"), "--register", str(reg),
               "--source", str(DATA / "guideline.txt"), "--out", str(out), "--offline")
    assert r.returncode != 0
    assert "resolve" in r.stderr and str(reg) in r.stderr and "Traceback" not in r.stderr
    log = {e["stage"]: e for e in read_log(out)}
    assert log["resolve"]["status"] == "failed" and str(reg) in log["resolve"]["message"]
    assert log["extract"]["status"] == "skipped" and not (out / "resolved.json").exists()


def test_pipeline_missing_inputs_name_the_stage(tmp_path):
    r = scribe("pipeline", "--transcript", str(tmp_path / "nope.txt"), "--register", str(DATA / "register.csv"),
               "--source", str(DATA / "guideline.txt"), "--out", str(tmp_path / "o1"), "--offline")
    assert r.returncode != 0 and "'extract'" in r.stderr and "nope.txt" in r.stderr
    r = scribe("pipeline", "--transcript", str(DATA / "transcript_01.txt"), "--register", str(DATA / "register.csv"),
               "--source", str(tmp_path / "nosource.txt"), "--out", str(tmp_path / "o2"), "--offline")
    assert r.returncode != 0 and "'knowledge'" in r.stderr
    log = {e["stage"]: e["status"] for e in read_log(tmp_path / "o2")}
    assert log["resolve"] == "ok" and log["knowledge"] == "failed"
