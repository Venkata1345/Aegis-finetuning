"""Tests for baselines/base.py — Pricing, Predictor.predict_batch default."""

from __future__ import annotations

from baselines.base import Pricing, Predictor, realign_to_input
from data.schema import Span


class TestPricing:
    def test_self_hosted_returns_none(self):
        p = Pricing()
        assert p.cost_for(100, 50) is None

    def test_partial_self_hosted_returns_none(self):
        # Only one rate set ⇒ still treated as self-hosted
        assert Pricing(input_per_1m=1.0).cost_for(100, 50) is None
        assert Pricing(output_per_1m=1.0).cost_for(100, 50) is None

    def test_missing_token_counts_returns_none(self):
        p = Pricing(input_per_1m=1.0, output_per_1m=2.0)
        assert p.cost_for(None, 100) is None
        assert p.cost_for(100, None) is None

    def test_correct_arithmetic(self):
        p = Pricing(input_per_1m=1.5, output_per_1m=3.0)
        # 1000 in × $1.5 + 200 out × $3.0 = 1500 + 600 = 2100; /1e6 = 0.0021
        assert abs(p.cost_for(1000, 200) - 0.0021) < 1e-9


class _CountingFake(Predictor):
    """Counts how many times _predict_impl is invoked."""

    def __init__(self) -> None:
        super().__init__(name="counter", pricing=Pricing())
        self.calls = 0

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        self.calls += 1
        return "[]", None, None


class TestPredictBatchDefault:
    def test_serial_default(self):
        p = _CountingFake()
        results = p.predict_batch(["a", "b", "c"])
        assert len(results) == 3
        assert p.calls == 3
        assert all(r.schema_valid for r in results)

    def test_empty_batch(self):
        p = _CountingFake()
        assert p.predict_batch([]) == []
        assert p.calls == 0


class TestRealignToInput:
    def _span(self, type_, start, end, text):
        return Span(type=type_, start=start, end=end, text=text)

    def test_correct_offset_unchanged(self):
        # Already correct → realignment leaves it as-is
        text = "Hello Alice"
        spans = [self._span("PERSON", 6, 11, "Alice")]
        out = realign_to_input(text, spans)
        assert out[0].start == 6 and out[0].end == 11 and out[0].text == "Alice"

    def test_wrong_offset_corrected(self):
        # Model emitted right text, wrong offsets — should snap to actual position
        text = "Hello Alice"
        spans = [self._span("PERSON", 0, 5, "Alice")]
        out = realign_to_input(text, spans)
        assert out[0].start == 6 and out[0].end == 11 and out[0].text == "Alice"

    def test_text_not_in_input_kept_as_is(self):
        # Model hallucinated text not in input — keep original; halluc metric catches it
        text = "Hello world"
        spans = [self._span("PERSON", 0, 3, "Bob")]
        out = realign_to_input(text, spans)
        assert out[0].start == 0 and out[0].end == 3 and out[0].text == "Bob"

    def test_multiple_occurrences_picks_closest(self):
        # "Bob" at positions 0 and 13. Model claimed start=12 → pick 13 not 0.
        text = "Bob said hi. Bob left."
        spans = [self._span("PERSON", 12, 15, "Bob")]
        out = realign_to_input(text, spans)
        assert out[0].start == 13 and out[0].end == 16

    def test_multiple_occurrences_picks_closest_other_direction(self):
        text = "Bob said hi. Bob left."
        spans = [self._span("PERSON", 1, 4, "Bob")]
        out = realign_to_input(text, spans)
        assert out[0].start == 0 and out[0].end == 3

    def test_phone_offset_off_by_significant_amount(self):
        # Real example from Aegis adversarial: model claimed start=10 for text at 11
        text = "Call me at 5 5 5 - 4 1 6 - 2 9 8 7."
        spans = [self._span("PHONE", 10, 27, "5 5 5 - 4 1 6 - 2 9 8 7")]
        out = realign_to_input(text, spans)
        assert out[0].start == 11
        assert out[0].end == 11 + len("5 5 5 - 4 1 6 - 2 9 8 7")
        # Aligned span text matches input slice
        assert text[out[0].start:out[0].end] == out[0].text
