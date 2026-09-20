"""Shape of note.json."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SECTIONS: tuple[str, ...] = (
    "chief_complaint",
    "history_of_presenting_illness",
    "review_of_systems",
    "past_medical_history",
    "past_surgical_history",
    "medication_history",
    "allergies",
    "family_history",
    "social_history",
    "vitals",
    "examination",
    "assessment",
    "plan",
)
NOT_STATED = "NOT_STATED"
CERTAINTIES = ("confirmed", "probable", "differential")
KIND_REJECTED = "considered_and_rejected"
ATTRIBUTIONS = ("patient", "doctor", "nurse", "companion")


class Span(BaseModel):
    ref: str
    text: str


class Element(BaseModel):
    """One traceable fact. Field order is the serialisation order."""

    value: str
    span: Span
    confidence: float = Field(ge=0.0, le=1.0)
    attribution: str | None = None
    certainty: Literal["confirmed", "probable", "differential"] | None = None
    kind: Literal["considered_and_rejected"] | None = None
    entity: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
