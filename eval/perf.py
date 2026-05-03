"""Performance measurement: latency percentiles, throughput, $/1k calls.

Pure functions over a list of Prediction objects (already collected by
the runner). Separating measurement from collection keeps perf reporting
deterministic — re-running the analysis on the same logs gives the
same numbers.
"""

from __future__ import annotations

import statistics

from baselines.base import Prediction, Pricing


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(round(p * (len(sorted_v) - 1)))
    return sorted_v[idx]


def latency_summary(predictions: list[Prediction]) -> dict:
    """p50/p99/mean/min/max latency in seconds."""
    latencies = [p.latency_s for p in predictions if p.error is None]
    if not latencies:
        return {"n": 0, "p50_s": 0.0, "p99_s": 0.0, "mean_s": 0.0, "min_s": 0.0, "max_s": 0.0}
    return {
        "n": len(latencies),
        "p50_s": statistics.median(latencies),
        "p99_s": _percentile(latencies, 0.99),
        "mean_s": statistics.mean(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
    }


def throughput_summary(predictions: list[Prediction]) -> dict:
    """Throughput in calls/sec (serial), based on accumulated latency.

    Wall-clock throughput requires a separate batch-size benchmark; this
    reports the steady-state assuming serial calls. Real per-batch numbers
    come from `benchmark_batch_throughput`.
    """
    latencies = [p.latency_s for p in predictions if p.error is None]
    if not latencies:
        return {"calls_per_sec": 0.0, "n": 0}
    total = sum(latencies)
    return {
        "n": len(latencies),
        "total_s": total,
        "calls_per_sec": len(latencies) / total if total > 0 else 0.0,
    }


def cost_summary(predictions: list[Prediction], pricing: Pricing) -> dict:
    """Aggregate token usage and dollar cost across a batch of predictions."""
    if pricing.input_per_1m is None or pricing.output_per_1m is None:
        return {
            "self_hosted": True,
            "n_calls": len(predictions),
            "note": "self-hosted — no per-token dollar cost",
        }

    n = len(predictions)
    counted = [p for p in predictions if p.input_tokens is not None and p.output_tokens is not None]
    total_in = sum(p.input_tokens or 0 for p in counted)
    total_out = sum(p.output_tokens or 0 for p in counted)
    total_usd = (
        total_in * pricing.input_per_1m + total_out * pricing.output_per_1m
    ) / 1_000_000

    return {
        "self_hosted": False,
        "n_calls": n,
        "n_with_token_counts": len(counted),
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "input_per_1m_usd": pricing.input_per_1m,
        "output_per_1m_usd": pricing.output_per_1m,
        "total_usd": total_usd,
        "per_call_usd": total_usd / len(counted) if counted else 0.0,
        "per_1k_usd": (total_usd / len(counted) * 1000) if counted else 0.0,
    }


def benchmark_batch_throughput(
    predictor,
    texts: list[str],
    batch_sizes: tuple[int, ...] = (1, 4, 16),
    warmup: int = 1,
) -> dict:
    """Measure wall-clock throughput at varying batch sizes.

    Uses predictor.predict_batch() — for predictors without native batching,
    this falls through to serial calls (so 4 vs 1 won't show speedup).
    """
    import time

    results = {}
    for bs in batch_sizes:
        if bs > len(texts):
            continue
        # Warmup
        for _ in range(warmup):
            predictor.predict_batch(texts[:bs])
        # Measure: process all texts in chunks of `bs`
        start = time.perf_counter()
        for i in range(0, len(texts), bs):
            predictor.predict_batch(texts[i : i + bs])
        elapsed = time.perf_counter() - start
        results[f"batch_{bs}"] = {
            "n": len(texts),
            "elapsed_s": elapsed,
            "throughput_per_sec": len(texts) / elapsed if elapsed > 0 else 0.0,
        }
    return results
