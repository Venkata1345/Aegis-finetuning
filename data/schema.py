"""Pydantic schema for PII span outputs.

Single source of truth for what a "valid" model output looks like. The
eval harness measures schema validity (% of outputs that parse here)
as a metric, so changes to this file change the contract for every
component — model, baselines, and the dataset converter all target it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

PIIType = Literal[
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "DOB",
    "GOV_ID",
    "FINANCIAL",
    "MEDICAL_ID",
    "IP_ADDRESS",
]

PII_TYPES: tuple[str, ...] = (
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "DOB",
    "GOV_ID",
    "FINANCIAL",
    "MEDICAL_ID",
    "IP_ADDRESS",
)


class Span(BaseModel):
    """A single PII detection. start/end are character offsets into the source text."""

    type: PIIType
    start: int = Field(ge=0)
    end: int
    text: str

    @model_validator(mode="after")
    def _check_offsets(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self


# Use to validate raw model output: SpanList.validate_json(generated_text)
SpanList = TypeAdapter(list[Span])
