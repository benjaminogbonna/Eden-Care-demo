# Tests for API endpoints.
import httpx
import pytest

from app.api.main import create_app
from app.core.config import Settings

ENVELOPE = {"success", "message", "data", "errors"}


def make_client(**overrides):
    settings = Settings(_env_file=None, gemini_api_key=None, **overrides)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings)), base_url="http://test")


@pytest.fixture()
async def client():
    async with make_client() as c:
        yield c


async def test_health_and_docs(client):
    r = await client.get("/health")
    assert r.status_code == 200 and set(r.json()) == ENVELOPE and r.json()["data"]["service"] == "scribe"
    assert (await client.get("/docs")).status_code == 200
    spec = (await client.get("/openapi.json")).json()
    assert {"/api/v1/extract", "/api/v1/validate", "/api/v1/resolve", "/api/v1/knowledge", "/api/v1/speech-eval", "/api/v1/pipeline"} <= set(spec["paths"])


async def test_extract_validate_resolve_flow(client, t1_text):
    r = await client.post("/api/v1/extract", json={"transcript": t1_text, "engine": "rules"})
    body = r.json()
    assert r.status_code == 200 and body["success"] and set(body) == ENVELOPE and body["data"]["engine"] == "rules"
    note = body["data"]["note"]
    r = await client.post("/api/v1/validate", json={"transcript": t1_text, "note": note})
    assert r.status_code == 200 and r.json()["data"] == {"valid": True}
    r = await client.post("/api/v1/resolve", json={"note": note})
    assert r.status_code == 200 and r.json()["data"]["resolved"]["assessment"][0]["code"] == "K29.7"


async def test_validate_rejection_uses_envelope(client, t1_text, note1):
    note1["vitals"][1]["value"] = "Pulse 67"
    r = await client.post("/api/v1/validate", json={"transcript": t1_text, "note": note1})
    body = r.json()
    assert r.status_code == 422 and body["success"] is False and set(body) == ENVELOPE
    assert body["errors"][0]["code"] == "number_not_in_span" and body["errors"][0]["field"] == "$.vitals[1]"


async def test_errors_are_enveloped(client):
    r = await client.post("/api/v1/extract", json={"transcript": "not a transcript"})
    assert r.status_code == 400 and set(r.json()) == ENVELOPE and r.json()["errors"][0]["code"] == "transcript_error"
    r = await client.post("/api/v1/extract", json={})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "body.transcript"
    r = await client.get("/api/v1/nothing-here")
    assert r.status_code == 404 and set(r.json()) == ENVELOPE and r.json()["success"] is False
    r = await client.post("/api/v1/resolve", json={"note": {}, "register_csv": "kind,code,name,synonyms\ndrug,A,B\n"})
    assert r.status_code == 400 and "expected 4 fields" in r.json()["message"]
    r = await client.post("/api/v1/extract", json={"transcript": "[00:00] DOCTOR: hi", "engine": "gemini"})
    assert r.status_code == 502 and "GEMINI_API_KEY" in r.json()["message"]


async def test_knowledge_and_speech_eval(client, t1_text, hyp1_text):
    r = await client.post("/api/v1/knowledge", json={})
    assert r.status_code == 200 and len(r.json()["data"]["rows"]) == 10
    r = await client.post("/api/v1/speech-eval", json={"reference": t1_text, "hypothesis": hyp1_text})
    assert r.status_code == 200 and r.json()["data"]["clinical_token_error_rate"] > 0


async def test_pipeline_ok_and_failure(client, t1_text):
    r = await client.post("/api/v1/pipeline", json={"transcript": t1_text, "engine": "rules"})
    assert r.status_code == 200 and r.json()["data"]["ok"] and len(r.json()["data"]["run_log"]) == 5
    r = await client.post("/api/v1/pipeline", json={"transcript": t1_text, "engine": "rules", "register_csv": "kind,code\n"})
    body = r.json()
    assert r.status_code == 422 and body["success"] is False and body["errors"][0]["field"] == "resolve"
    assert {e["stage"]: e["status"] for e in body["data"]["run_log"]}["resolve"] == "failed"


async def test_api_key_protection(t1_text):
    async with make_client(api_key="s3cret") as c:
        assert (await c.get("/health")).status_code == 200
        r = await c.post("/api/v1/extract", json={"transcript": t1_text, "engine": "rules"})
        assert r.status_code == 401 and set(r.json()) == ENVELOPE
        r = await c.post("/api/v1/extract", json={"transcript": t1_text, "engine": "rules"}, headers={"X-API-Key": "wrong"})
        assert r.status_code == 401
        r = await c.post("/api/v1/extract", json={"transcript": t1_text, "engine": "rules"}, headers={"X-API-Key": "s3cret"})
        assert r.status_code == 200


async def test_body_size_limit(t1_text):
    async with make_client(max_body_bytes=200) as c:
        r = await c.post("/api/v1/extract", json={"transcript": t1_text})
        assert r.status_code == 413 and r.json()["errors"][0]["code"] == "body_too_large"
