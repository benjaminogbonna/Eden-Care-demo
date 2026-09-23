from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def t1_text() -> str:
    return (DATA / "transcript_01.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def hyp1_text() -> str:
    return (DATA / "hyp_01.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def companion_text() -> str:
    return (DATA / "transcript_companion_test.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def register_path() -> Path:
    return DATA / "register.csv"


@pytest.fixture(scope="session")
def guideline_text() -> str:
    return (DATA / "guideline.txt").read_text(encoding="utf-8")


@pytest.fixture()
def note1(t1_text):
    import asyncio
    from app.services.extraction.service import extract_note
    return asyncio.run(extract_note(t1_text, engine="rules")).note


@pytest.fixture()
def transcript1(t1_text):
    from app.core.transcript import parse_transcript
    return parse_transcript(t1_text)


@pytest.fixture()
def no_key_settings():
    from app.core.config import Settings
    return Settings(_env_file=None, gemini_api_key=None)


@pytest.fixture()
def tampered():
    def make(note, mutate):
        n = copy.deepcopy(note)
        mutate(n)
        return n
    return make
