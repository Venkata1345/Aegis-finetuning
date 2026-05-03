"""Predictor protocol — what every baseline (and Aegis itself) implements.

Subclass `Predictor` and implement `_predict_impl(text)`. The base class
handles timing, code-fence stripping, JSON parsing, and exception
catching, so concrete predictors only have to make their call.

Pricing is in USD per 1M tokens. Self-hosted predictors leave it as
None — `cost_for(...)` returns None and the eval harness reports
"self-hosted" for those rows.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import ValidationError

from data.schema import Span, SpanList


@dataclass
class Pricing:
    """USD per 1M tokens. None ⇒ self-hosted (no per-call dollar cost)."""

    input_per_1m: float | None = None
    output_per_1m: float | None = None

    def cost_for(self, in_tok: int | None, out_tok: int | None) -> float | None:
        if self.input_per_1m is None or self.output_per_1m is None:
            return None
        if in_tok is None or out_tok is None:
            return None
        return (in_tok * self.input_per_1m + out_tok * self.output_per_1m) / 1_000_000


@dataclass
class Prediction:
    """Result of a single prediction call.

    `spans` is the parsed-and-validated list (empty if schema-invalid or error).
    `raw` is the post-fence-strip output text, used for schema_validity_rate.
    """

    spans: list[Span] = field(default_factory=list)
    raw: str = ""
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    schema_valid: bool = True
    error: str | None = None


def strip_code_fences(s: str) -> str:
    """Remove leading/trailing ```json ... ``` (or plain ```) fences if present.

    Some models emit fences despite "JSON only" instructions; we strip
    deterministically so downstream parsing has a consistent input.
    """
    s = s.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    # Drop opening fence line
    lines = lines[1:]
    # Drop trailing closing fence + any blank lines
    while lines and lines[-1].strip() in ("```", ""):
        lines.pop()
    return "\n".join(lines).strip()


class Predictor(ABC):
    """Abstract base. Subclass and implement `_predict_impl`."""

    def __init__(self, name: str, pricing: Pricing | None = None) -> None:
        self.name = name
        self.pricing = pricing or Pricing()

    @abstractmethod
    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        """Return (raw_model_output, input_tokens, output_tokens).

        Token counts may be None when not measurable (e.g. rule-based predictors).
        Raise on transport errors; the base class catches and records them.
        """

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        try:
            raw, in_tok, out_tok = self._predict_impl(text)
        except Exception as e:  # noqa: BLE001 — we want every failure surface to flow through
            return Prediction(
                spans=[],
                raw="",
                latency_s=time.perf_counter() - start,
                schema_valid=False,
                error=f"{type(e).__name__}: {e}",
            )
        latency = time.perf_counter() - start

        cleaned = strip_code_fences(raw)
        try:
            spans = SpanList.validate_json(cleaned)
            valid = True
        except (ValidationError, json.JSONDecodeError, ValueError):
            spans = []
            valid = False

        return Prediction(
            spans=spans,
            raw=cleaned,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            schema_valid=valid,
        )

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Default: serial. Subclasses with native batching should override."""
        return [self.predict(t) for t in texts]
