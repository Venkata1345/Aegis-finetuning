"""Pure metric functions. The foundation of every claim in the README.

Conventions (deliberate; departures should be discussed):

- A **span** is identified by the triple (type, start, end). Two spans with
  the same triple are treated as the same span; duplicate predictions
  collapse via set semantics. Real-world model outputs rarely duplicate, but
  if they do, dedup is the standard NER convention.

- **Strict match**: predicted span matches gold iff (type, start, end) match
  exactly. Type-only mismatches with identical offsets count as BOTH a
  per-type FN (for the gold's type) AND a per-type FP (for the pred's
  type), so the overall counts are the sum of per-type counts.

- **Partial match (IoU)**: predicted span matches gold if same type AND
  span overlap (intersection / union) >= threshold (default 0.5). Each pred
  matches at most one gold (greedy by best IoU); each gold matches at most
  one pred. Cross-type matches never count, even with full overlap — type
  mismatch is a hard error.

- **Hallucination**: predicted span where text[start:end] != span.text, or
  offsets out of bounds. Schema-valid output but the model invented offsets
  that don't anchor in the input. Distinct from schema-invalid output
  (caught by schema_validity_rate).

- **Schema validity**: % of raw model outputs that parse via the Pydantic
  SpanList. A separate metric because some baselines (Presidio) emit
  programmatic spans that always validate; for these, schema_validity is
  N/A — caller decides whether to compute it.

All functions are pure: take typed inputs, return dicts. No I/O, no
mutation. Runner / harness layers above are responsible for orchestration.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import ValidationError

from data.schema import PII_TYPES, Span, SpanList


@dataclass
class Instance:
    """One example with parsed gold and predicted spans."""

    text: str
    gold: list[Span]
    pred: list[Span]


@dataclass
class RawOutput:
    """One example with the model's RAW string output, before parsing.

    Used for schema_validity_rate. `gold` is included for symmetry with
    Instance but is not used by schema validity itself.
    """

    text: str
    gold: list[Span]
    raw: str
    parsed: list[Span] | None = None  # populated by parse_with_validity()


# --- core helpers ----------------------------------------------------------


def _key(s: Span) -> tuple[str, int, int]:
    return (s.type, s.start, s.end)


def _iou(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _format_counts(c: _Counts) -> dict:
    p, r, f1 = _prf1(c.tp, c.fp, c.fn)
    return {"tp": c.tp, "fp": c.fp, "fn": c.fn, "precision": p, "recall": r, "f1": f1}


# --- strict F1 -------------------------------------------------------------


def strict_f1(instances: list[Instance]) -> dict:
    """Strict span F1: exact (type, start, end) match required.

    Returns:
        {
            "overall":  {tp, fp, fn, precision, recall, f1},
            "per_type": {<TYPE>: {tp, fp, fn, precision, recall, f1}, ...},
        }
    """
    overall = _Counts()
    per_type: dict[str, _Counts] = defaultdict(_Counts)

    for inst in instances:
        gold_keys = {_key(s) for s in inst.gold}
        pred_keys = {_key(s) for s in inst.pred}

        # per-type
        gold_by_type: dict[str, set] = defaultdict(set)
        pred_by_type: dict[str, set] = defaultdict(set)
        for s in inst.gold:
            gold_by_type[s.type].add(_key(s))
        for s in inst.pred:
            pred_by_type[s.type].add(_key(s))
        for t in PII_TYPES:
            g, p = gold_by_type[t], pred_by_type[t]
            per_type[t].tp += len(g & p)
            per_type[t].fp += len(p - g)
            per_type[t].fn += len(g - p)

        # overall
        overall.tp += len(gold_keys & pred_keys)
        overall.fp += len(pred_keys - gold_keys)
        overall.fn += len(gold_keys - pred_keys)

    return {
        "overall": _format_counts(overall),
        "per_type": {t: _format_counts(per_type[t]) for t in PII_TYPES},
    }


# --- partial F1 (IoU >= threshold) -----------------------------------------


def partial_f1(instances: list[Instance], iou_threshold: float = 0.5) -> dict:
    """Partial-span F1: TP if same type AND IoU >= threshold.

    Greedy matching: each pred matches at most the best-IoU available gold;
    each gold matches at most one pred. Cross-type matches never count.
    """
    overall = _Counts()
    per_type: dict[str, _Counts] = defaultdict(_Counts)

    for inst in instances:
        gold_by_type: dict[str, list[Span]] = defaultdict(list)
        pred_by_type: dict[str, list[Span]] = defaultdict(list)
        for s in inst.gold:
            gold_by_type[s.type].append(s)
        for s in inst.pred:
            pred_by_type[s.type].append(s)

        all_types = set(gold_by_type) | set(pred_by_type)
        for t in all_types:
            golds = gold_by_type[t]
            preds = pred_by_type[t]
            matched_g: set[int] = set()
            tp = 0
            # Sort preds by best available IoU descending — greedy needs an
            # order that maximises total matches; sorting by max IoU first
            # is a good heuristic and matches common NER eval scripts.
            scored = []
            for p in preds:
                best_iou = max((_iou(p, g) for g in golds), default=0.0)
                scored.append((best_iou, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, p in scored:
                best_idx, best_iou = -1, 0.0
                for gi, g in enumerate(golds):
                    if gi in matched_g:
                        continue
                    iou = _iou(p, g)
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                if best_idx >= 0 and best_iou >= iou_threshold:
                    matched_g.add(best_idx)
                    tp += 1
            fp = len(preds) - tp
            fn = len(golds) - tp
            per_type[t].tp += tp
            per_type[t].fp += fp
            per_type[t].fn += fn
            overall.tp += tp
            overall.fp += fp
            overall.fn += fn

    return {
        "overall": _format_counts(overall),
        "per_type": {t: _format_counts(per_type[t]) for t in PII_TYPES},
        "iou_threshold": iou_threshold,
    }


# --- schema validity ------------------------------------------------------


def parse_with_validity(raw_outputs: list[RawOutput]) -> dict:
    """Try to parse each raw output via SpanList. Mutates `parsed` in place.

    Returns the schema-validity report; the populated RawOutput.parsed
    fields can be passed to downstream Instance construction.
    """
    valid = 0
    failures: list[dict] = []
    for r in raw_outputs:
        try:
            spans = SpanList.validate_json(r.raw)
            r.parsed = spans
            valid += 1
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            r.parsed = None
            # Truncate raw to keep the report compact
            failures.append({"raw_excerpt": r.raw[:200], "error": type(e).__name__})
    total = len(raw_outputs)
    return {
        "total": total,
        "valid": valid,
        "n_failed": total - valid,
        "rate": valid / total if total > 0 else 0.0,
        "failures": failures[:10],  # cap for readability
    }


# --- hallucination --------------------------------------------------------


def correctly_empty_rate(instances: list[Instance]) -> dict:
    """For instances with EMPTY gold, did the predictor also predict nothing?

    Strict F1 is meaningless for no-PII inputs (no TP possible, P/R undefined),
    so this is the dedicated metric for negative controls (`reference_no_value`,
    `zero_pii_control`, near-miss cases where the gold is `[]`).

    Two complementary numbers:
      - `rate`: fraction of empty-gold instances where pred was also empty
      - `avg_fp_per_empty`: average FP count when the gold was empty
        (distinguishes "rarely correct but only 1 stray FP" from "rarely
        correct and 10 stray FPs per input")
    """
    empty_gold = [i for i in instances if not i.gold]
    if not empty_gold:
        return {
            "empty_gold_n": 0,
            "correctly_empty_n": 0,
            "rate": 0.0,
            "avg_fp_per_empty": 0.0,
        }
    correctly = sum(1 for i in empty_gold if not i.pred)
    total_fps = sum(len(i.pred) for i in empty_gold)
    return {
        "empty_gold_n": len(empty_gold),
        "correctly_empty_n": correctly,
        "rate": correctly / len(empty_gold),
        "avg_fp_per_empty": total_fps / len(empty_gold),
    }


def hallucination_rate(instances: list[Instance]) -> dict:
    """% of predicted spans whose offsets don't anchor to the input text.

    Hallucinated = text[start:end] != span.text, OR offsets out of bounds.
    The model emitted schema-valid spans but invented their location.
    """
    total = 0
    hallucinated = 0
    examples: list[dict] = []
    for inst in instances:
        for s in inst.pred:
            total += 1
            if (
                s.start < 0
                or s.end > len(inst.text)
                or inst.text[s.start : s.end] != s.text
            ):
                hallucinated += 1
                if len(examples) < 5:
                    examples.append({
                        "claimed_text": s.text,
                        "actual_text": (
                            inst.text[s.start : s.end]
                            if 0 <= s.start and s.end <= len(inst.text)
                            else "<out-of-bounds>"
                        ),
                        "span": {"type": s.type, "start": s.start, "end": s.end},
                    })
    return {
        "total_predictions": total,
        "hallucinated": hallucinated,
        "rate": hallucinated / total if total > 0 else 0.0,
        "examples": examples,
    }
