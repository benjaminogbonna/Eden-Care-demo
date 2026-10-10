# Tests for the Gemini extraction path. 
"""Gemini path, exercised with an injected fake client (no network). The real SDK types are used for the config."""
import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.transcript import parse_transcript
from app.domain.note import SECTIONS
from app.core.errors import ExtractionError
from app.services.extraction.gemini import GeminiExtractor, ground, prompt_hash
from app.services.extraction.service import extract_note
from app.services.validation import validate_note

