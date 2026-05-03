"""Eval runner: take a Predictor and a JSONL dataset, return all metrics.

This is the integration glue between baselines/ and eval/metrics+perf.
Pure-ish: I/O for reading the dataset and (optionally) writing per-instance
logs for error analysis; no other state.

Dataset JSONL format (matches data/eval/heldout.jsonl and adversarial.jsonl):
  {"id": ..., "text": ..., "spans": [...], "category": ...}

The optional `category` field powers the by_category breakdown — the
adversarial set uses it heavily so we can report metrics per adversarial
sub-type.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.base import Prediction, Predictor
from data.schema import Span
from eval.metrics import (
    Instance,
    RawOutput,
    correctly_empty_rate,
    hallucination_rate,
    parse_with_validity,
    partial_f1,
    strict_f1,
)
from eval.perf import cost_summary, latency_summary


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(
    predictor: Predictor,
    dataset_path: Path,
    *,
    log_path: Path | None = None,
    max_examples: int | None = None,
    by_category: bool = False,
) -> dict:
    """Run the predictor against the dataset and compute the full metric set.

    Args:
        predictor: a Predictor instance.
        dataset_path: JSONL file with rows of {text, spans, ...}.
        log_path: if given, write per-instance results to this JSONL for
            error analysis (gold, pred, raw, latency, error). Useful and small.
        max_examples: cap rows processed (for smoke tests).
        by_category: also compute strict_f1 + partial_f1 grouped by row's
            'category' field — the adversarial set sets this.

    Returns: a dict with strict_f1, partial_f1, schema_validity, hallucination,
    perf (latency + cost), and (if by_category) per-category breakdown.
    """
    rows = load_dataset(Path(dataset_path))
    if max_examples is not None:
        rows = rows[:max_examples]

    instances: list[Instance] = []
    raws: list[RawOutput] = []
    predictions: list[Prediction] = []
    log_records: list[dict] = []

    for row in rows:
        text = row["text"]
        gold = [Span(**s) for s in row.get("spans", [])]
        pred_obj = predictor.predict(text)
        predictions.append(pred_obj)
        instances.append(Instance(text=text, gold=gold, pred=pred_obj.spans))
        raws.append(RawOutput(text=text, gold=gold, raw=pred_obj.raw))
        if log_path is not None:
            log_records.append({
                "id": row.get("id"),
                "category": row.get("category"),
                "text_len": len(text),
                "gold": [s.model_dump() for s in gold],
                "pred": [s.model_dump() for s in pred_obj.spans],
                "raw": pred_obj.raw,
                "latency_s": pred_obj.latency_s,
                "input_tokens": pred_obj.input_tokens,
                "output_tokens": pred_obj.output_tokens,
                "schema_valid": pred_obj.schema_valid,
                "error": pred_obj.error,
            })

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            for r in log_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    result: dict = {
        "predictor": predictor.name,
        "dataset": str(dataset_path),
        "n_examples": len(rows),
        "strict_f1": strict_f1(instances),
        "partial_f1": partial_f1(instances),
        "correctly_empty": correctly_empty_rate(instances),
        "schema_validity": parse_with_validity(raws),
        "hallucination": hallucination_rate(instances),
        "perf": {
            "latency": latency_summary(predictions),
            "cost": cost_summary(predictions, predictor.pricing),
        },
        "errors": [p.error for p in predictions if p.error is not None],
    }

    if by_category:
        groups: dict[str, list[Instance]] = {}
        for row, inst in zip(rows, instances, strict=True):
            cat = row.get("category", "unknown")
            groups.setdefault(cat, []).append(inst)
        cat_results = {}
        for cat, cat_instances in groups.items():
            cat_results[cat] = {
                "n": len(cat_instances),
                "strict_f1": strict_f1(cat_instances)["overall"],
                "partial_f1": partial_f1(cat_instances)["overall"],
                "correctly_empty": correctly_empty_rate(cat_instances),
            }
        result["by_category"] = cat_results

    return result
