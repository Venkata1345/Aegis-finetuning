"""Tests for eval/perf.py — latency / throughput / cost / batch benchmark."""

from __future__ import annotations

from baselines.base import Prediction, Pricing
from eval.perf import (
    _percentile,
    benchmark_batch_throughput,
    cost_summary,
    latency_summary,
    throughput_summary,
)


def _pred(latency_s: float = 0.1, error: str | None = None,
          in_tok: int | None = None, out_tok: int | None = None) -> Prediction:
    return Prediction(
        latency_s=latency_s, error=error,
        input_tokens=in_tok, output_tokens=out_tok,
    )


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 0.5) == 0.0

    def test_single(self):
        assert _percentile([1.0], 0.5) == 1.0
        assert _percentile([1.0], 0.99) == 1.0

    def test_p50_p99(self):
        vs = [float(i) for i in range(1, 101)]  # 1..100
        # int(round(0.5 * 99)) = round(49.5) = 50 (banker's rounding) → values[50] = 51.0.
        # Either-side-of-median is acceptable; just verify reasonable behavior.
        assert _percentile(vs, 0.50) == 51.0
        # int(round(0.99 * 99)) = round(98.01) = 98 → values[98] = 99.0
        assert _percentile(vs, 0.99) == 99.0


class TestLatencySummary:
    def test_empty_returns_zeros(self):
        out = latency_summary([])
        assert out == {"n": 0, "p50_s": 0.0, "p99_s": 0.0, "mean_s": 0.0, "min_s": 0.0, "max_s": 0.0}

    def test_excludes_errored(self):
        preds = [_pred(0.1), _pred(0.2), _pred(99.0, error="oops")]
        out = latency_summary(preds)
        assert out["n"] == 2
        # 99.0 with error excluded ⇒ max should be 0.2
        assert out["max_s"] == 0.2

    def test_basic(self):
        preds = [_pred(s) for s in [0.1, 0.2, 0.3, 0.4, 0.5]]
        out = latency_summary(preds)
        assert out["n"] == 5
        assert out["min_s"] == 0.1
        assert out["max_s"] == 0.5
        assert out["mean_s"] == 0.3
        assert out["p50_s"] == 0.3


class TestThroughputSummary:
    def test_empty(self):
        out = throughput_summary([])
        assert out == {"calls_per_sec": 0.0, "n": 0}

    def test_basic(self):
        preds = [_pred(s) for s in [0.5, 0.5, 1.0]]
        out = throughput_summary(preds)
        assert out["n"] == 3
        assert out["total_s"] == 2.0
        assert abs(out["calls_per_sec"] - 1.5) < 1e-9


class TestCostSummary:
    def test_self_hosted(self):
        out = cost_summary([_pred(), _pred()], Pricing())
        assert out["self_hosted"] is True
        assert out["n_calls"] == 2

    def test_priced(self):
        preds = [_pred(in_tok=500, out_tok=100), _pred(in_tok=500, out_tok=100)]
        pricing = Pricing(input_per_1m=2.0, output_per_1m=4.0)
        out = cost_summary(preds, pricing)
        # per call: 500 × $2 + 100 × $4 = 1000 + 400 = 1400; /1e6 = $0.0014
        # 2 calls ⇒ total $0.0028
        assert abs(out["total_usd"] - 0.0028) < 1e-9
        assert abs(out["per_call_usd"] - 0.0014) < 1e-9
        assert abs(out["per_1k_usd"] - 1.4) < 1e-9
        assert out["n_with_token_counts"] == 2

    def test_priced_with_missing_tokens(self):
        # One call with token counts, one without
        preds = [_pred(in_tok=1000, out_tok=200), _pred(in_tok=None, out_tok=None)]
        pricing = Pricing(input_per_1m=1.0, output_per_1m=2.0)
        out = cost_summary(preds, pricing)
        # Only the first call counted: 1000×$1 + 200×$2 = 1400; /1e6 = $0.0014
        assert out["n_calls"] == 2
        assert out["n_with_token_counts"] == 1
        assert abs(out["total_usd"] - 0.0014) < 1e-9


class _SerialFake:
    """Minimal stand-in for a Predictor — only needs predict_batch for benchmarks."""

    def __init__(self) -> None:
        self.calls = 0

    def predict_batch(self, texts):
        self.calls += len(texts)
        return [Prediction(latency_s=0.001) for _ in texts]


class TestBenchmarkBatchThroughput:
    def test_runs_each_batch_size(self):
        p = _SerialFake()
        results = benchmark_batch_throughput(p, ["a", "b", "c", "d"], batch_sizes=(1, 2), warmup=1)
        assert "batch_1" in results
        assert "batch_2" in results
        assert results["batch_1"]["n"] == 4
        assert results["batch_2"]["n"] == 4

    def test_skips_oversize_batch(self):
        p = _SerialFake()
        results = benchmark_batch_throughput(p, ["a", "b"], batch_sizes=(1, 16), warmup=0)
        assert "batch_1" in results
        assert "batch_16" not in results  # batch > len(texts) skipped
