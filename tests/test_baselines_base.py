"""Tests for baselines/base.py — Pricing, Predictor.predict_batch default."""

from __future__ import annotations

from baselines.base import Pricing, Predictor


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
