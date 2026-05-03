"""Unit tests for eval/metrics.py — the foundation of every README claim.

Coverage:
  - _iou: full overlap, no overlap, partial, adjacent
  - _prf1: zeros, perfect, asymmetric
  - strict_f1: empty, perfect, type-mismatch, offset-mismatch, partial overlap
  - partial_f1: IoU threshold behavior, greedy matching, cross-type rejection
  - parse_with_validity: valid array, invalid JSON, schema-violating shape
  - hallucination_rate: clean, invented offsets, out-of-bounds
"""

from __future__ import annotations

import json

import pytest

from data.schema import Span
from eval.metrics import (
    Instance,
    RawOutput,
    _iou,
    _prf1,
    correctly_empty_rate,
    hallucination_rate,
    parse_with_validity,
    partial_f1,
    strict_f1,
)


# ---------------------------------------------------------------- _iou


def _span(type_: str, start: int, end: int, text: str | None = None) -> Span:
    """Build a Span. If text is None, fill with 'x' chars to match length."""
    if text is None:
        text = "x" * (end - start)
    return Span(type=type_, start=start, end=end, text=text)


class TestIoU:
    def test_full_overlap(self):
        a = _span("PERSON", 0, 10)
        b = _span("PERSON", 0, 10)
        assert _iou(a, b) == 1.0

    def test_no_overlap(self):
        a = _span("PERSON", 0, 5)
        b = _span("PERSON", 10, 15)
        assert _iou(a, b) == 0.0

    def test_adjacent(self):
        # touching but not overlapping
        a = _span("PERSON", 0, 5)
        b = _span("PERSON", 5, 10)
        assert _iou(a, b) == 0.0

    def test_partial(self):
        a = _span("PERSON", 0, 10)
        b = _span("PERSON", 5, 15)
        # intersection = 5, union = 15, IoU = 1/3
        assert abs(_iou(a, b) - 1 / 3) < 1e-9

    def test_one_contains_other(self):
        a = _span("PERSON", 0, 10)
        b = _span("PERSON", 2, 8)
        # intersection = 6, union = 10, IoU = 0.6
        assert abs(_iou(a, b) - 0.6) < 1e-9


# ---------------------------------------------------------------- _prf1


class TestPRF1:
    def test_all_zeros(self):
        assert _prf1(0, 0, 0) == (0.0, 0.0, 0.0)

    def test_perfect(self):
        assert _prf1(10, 0, 0) == (1.0, 1.0, 1.0)

    def test_only_fp(self):
        # no true positives, only false positives
        p, r, f1 = _prf1(0, 5, 0)
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_only_fn(self):
        p, r, f1 = _prf1(0, 0, 5)
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_asymmetric(self):
        # 6 TP, 2 FP, 4 FN -> p=6/8=0.75, r=6/10=0.6
        p, r, f1 = _prf1(6, 2, 4)
        assert abs(p - 0.75) < 1e-9
        assert abs(r - 0.6) < 1e-9
        # f1 = 2*0.75*0.6/(0.75+0.6) = 0.9/1.35 ≈ 0.6667
        assert abs(f1 - 2 / 3) < 1e-9


# ---------------------------------------------------------------- strict_f1


class TestStrictF1:
    def test_empty(self):
        out = strict_f1([Instance(text="hello", gold=[], pred=[])])
        assert out["overall"]["tp"] == 0
        assert out["overall"]["fp"] == 0
        assert out["overall"]["fn"] == 0
        assert out["overall"]["f1"] == 0.0

    def test_perfect(self):
        spans = [_span("PERSON", 0, 5, "Alice")]
        out = strict_f1([Instance(text="Alice", gold=spans, pred=spans)])
        assert out["overall"]["f1"] == 1.0
        assert out["per_type"]["PERSON"]["f1"] == 1.0

    def test_type_mismatch_double_counts(self):
        # Same offsets, different types: one FN (PERSON) and one FP (EMAIL)
        gold = [_span("PERSON", 0, 5, "Alice")]
        pred = [_span("EMAIL", 0, 5, "Alice")]
        out = strict_f1([Instance(text="Alice", gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 0
        assert out["overall"]["fp"] == 1
        assert out["overall"]["fn"] == 1
        # Per-type sum equals overall
        assert out["per_type"]["PERSON"]["fn"] == 1
        assert out["per_type"]["EMAIL"]["fp"] == 1

    def test_offset_mismatch_no_credit(self):
        # Different offsets, same type → strict gives 0 credit
        gold = [_span("PERSON", 0, 5, "Alice")]
        pred = [_span("PERSON", 0, 4, "Alic")]
        out = strict_f1([Instance(text="Alice", gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 0
        assert out["overall"]["fp"] == 1
        assert out["overall"]["fn"] == 1

    def test_aggregation_across_instances(self):
        i1 = Instance(
            text="x" * 20,
            gold=[_span("PERSON", 0, 5)],
            pred=[_span("PERSON", 0, 5)],
        )
        i2 = Instance(
            text="x" * 20,
            gold=[_span("EMAIL", 10, 20)],
            pred=[],  # missed
        )
        out = strict_f1([i1, i2])
        assert out["overall"]["tp"] == 1
        assert out["overall"]["fp"] == 0
        assert out["overall"]["fn"] == 1
        assert out["per_type"]["PERSON"]["tp"] == 1
        assert out["per_type"]["EMAIL"]["fn"] == 1
        # F1 = 2*1*0.5/1.5 = 2/3
        assert abs(out["overall"]["f1"] - 2 / 3) < 1e-9

    def test_duplicate_predictions_dedup(self):
        # Convention: duplicate (type, start, end) collapses
        gold = [_span("PERSON", 0, 5)]
        pred = [_span("PERSON", 0, 5), _span("PERSON", 0, 5)]
        out = strict_f1([Instance(text="x" * 10, gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 1
        assert out["overall"]["fp"] == 0


# ---------------------------------------------------------------- partial_f1


class TestPartialF1:
    def test_perfect_overlap_counts(self):
        gold = [_span("PERSON", 0, 10)]
        pred = [_span("PERSON", 0, 10)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        assert out["overall"]["f1"] == 1.0

    def test_iou_below_threshold_misses(self):
        # IoU = 1/4 < 0.5
        gold = [_span("PERSON", 0, 4)]
        pred = [_span("PERSON", 3, 10)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        # IoU = 1 / (4+7-1) = 1/10 = 0.1 actually. Just confirm 0 TP.
        assert out["overall"]["tp"] == 0

    def test_iou_above_threshold_hits(self):
        # gold [0,10], pred [2,10] -> intersection=8, union=10, IoU=0.8
        gold = [_span("PERSON", 0, 10)]
        pred = [_span("PERSON", 2, 10)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 1

    def test_threshold_boundary_inclusive(self):
        # gold [0,10], pred [5,10] -> intersection=5, union=10, IoU=0.5
        gold = [_span("PERSON", 0, 10)]
        pred = [_span("PERSON", 5, 10)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 1, "IoU == threshold should match"

    def test_cross_type_no_credit_even_with_overlap(self):
        gold = [_span("PERSON", 0, 10)]
        pred = [_span("EMAIL", 0, 10)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 0
        assert out["overall"]["fp"] == 1
        assert out["overall"]["fn"] == 1

    def test_greedy_one_pred_per_gold(self):
        # Two preds overlapping the same gold; only one matches.
        gold = [_span("PERSON", 0, 10)]
        pred = [_span("PERSON", 0, 10), _span("PERSON", 1, 9)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 1
        assert out["overall"]["fp"] == 1

    def test_greedy_one_gold_per_pred(self):
        # Two golds, one pred — pred matches the better-IoU gold
        gold = [_span("PERSON", 0, 10), _span("PERSON", 0, 8)]
        pred = [_span("PERSON", 0, 10)]
        out = partial_f1([Instance(text="x" * 20, gold=gold, pred=pred)])
        assert out["overall"]["tp"] == 1
        assert out["overall"]["fn"] == 1

    def test_custom_threshold(self):
        # IoU = 0.4. With threshold 0.5: miss. With threshold 0.4: hit.
        gold = [_span("PERSON", 0, 10)]
        pred = [_span("PERSON", 4, 10)]  # intersection=6, union=10, IoU=0.6
        out_strict = partial_f1(
            [Instance(text="x" * 20, gold=gold, pred=pred)],
            iou_threshold=0.7,
        )
        assert out_strict["overall"]["tp"] == 0  # 0.6 < 0.7
        out_loose = partial_f1(
            [Instance(text="x" * 20, gold=gold, pred=pred)],
            iou_threshold=0.5,
        )
        assert out_loose["overall"]["tp"] == 1


# ---------------------------------------------------------------- parse_with_validity


class TestSchemaValidity:
    def test_all_valid(self):
        raws = [
            RawOutput(text="x", gold=[], raw='[]'),
            RawOutput(
                text="Alice", gold=[],
                raw='[{"type":"PERSON","start":0,"end":5,"text":"Alice"}]',
            ),
        ]
        out = parse_with_validity(raws)
        assert out["valid"] == 2
        assert out["rate"] == 1.0
        assert raws[0].parsed == []
        assert raws[1].parsed is not None and len(raws[1].parsed) == 1

    def test_invalid_json(self):
        raws = [RawOutput(text="x", gold=[], raw="not json {")]
        out = parse_with_validity(raws)
        assert out["valid"] == 0
        assert out["n_failed"] == 1

    def test_valid_json_wrong_shape(self):
        # JSON parses but doesn't match list[Span]
        raws = [RawOutput(text="x", gold=[], raw='{"spans": []}')]
        out = parse_with_validity(raws)
        assert out["valid"] == 0

    def test_valid_json_invalid_type(self):
        raws = [
            RawOutput(
                text="Alice", gold=[],
                raw='[{"type":"NAME","start":0,"end":5,"text":"Alice"}]',
            ),
        ]
        out = parse_with_validity(raws)
        # NAME is not in the PIIType literal — should fail
        assert out["valid"] == 0

    def test_valid_json_inverted_offsets(self):
        # end < start violates the model_validator
        raws = [
            RawOutput(
                text="x", gold=[],
                raw='[{"type":"PERSON","start":10,"end":5,"text":"x"}]',
            ),
        ]
        out = parse_with_validity(raws)
        assert out["valid"] == 0

    def test_mixed(self):
        raws = [
            RawOutput(text="x", gold=[], raw='[]'),
            RawOutput(text="x", gold=[], raw='not json'),
            RawOutput(text="x", gold=[], raw='[]'),
        ]
        out = parse_with_validity(raws)
        assert out["valid"] == 2
        assert out["total"] == 3
        assert abs(out["rate"] - 2 / 3) < 1e-9


# ---------------------------------------------------------------- hallucination_rate


class TestHallucinationRate:
    def test_clean_predictions(self):
        text = "Alice is here"
        gold = []
        pred = [_span("PERSON", 0, 5, "Alice")]
        # text[0:5] == "Alice" ✓
        out = hallucination_rate([Instance(text=text, gold=gold, pred=pred)])
        assert out["hallucinated"] == 0
        assert out["rate"] == 0.0

    def test_invented_text_field(self):
        text = "Alice is here"
        # span claims "Bob" but text[0:5] == "Alice"
        bad = Span(type="PERSON", start=0, end=5, text="Bob X")
        out = hallucination_rate([Instance(text=text, gold=[], pred=[bad])])
        assert out["hallucinated"] == 1
        assert out["rate"] == 1.0

    def test_out_of_bounds(self):
        text = "short"
        bad = Span(type="PERSON", start=10, end=15, text="aaaaa")
        out = hallucination_rate([Instance(text=text, gold=[], pred=[bad])])
        assert out["hallucinated"] == 1

    def test_mixed(self):
        text = "Alice and Bob"
        good = Span(type="PERSON", start=0, end=5, text="Alice")
        bad = Span(type="PERSON", start=10, end=13, text="Carol")  # actual is "Bob"
        out = hallucination_rate(
            [Instance(text=text, gold=[], pred=[good, bad])],
        )
        assert out["total_predictions"] == 2
        assert out["hallucinated"] == 1
        assert out["rate"] == 0.5

    def test_no_predictions(self):
        out = hallucination_rate(
            [Instance(text="anything", gold=[], pred=[])],
        )
        assert out["total_predictions"] == 0
        assert out["rate"] == 0.0


# ---------------------------------------------------------------- correctly_empty_rate


class TestCorrectlyEmptyRate:
    def test_no_empty_gold_returns_zeros(self):
        # Every instance has gold spans
        inst = Instance(
            text="Alice", gold=[_span("PERSON", 0, 5)], pred=[]
        )
        out = correctly_empty_rate([inst])
        assert out["empty_gold_n"] == 0
        assert out["correctly_empty_n"] == 0
        assert out["rate"] == 0.0
        assert out["avg_fp_per_empty"] == 0.0

    def test_all_correctly_empty(self):
        inst = [
            Instance(text="x", gold=[], pred=[]),
            Instance(text="y", gold=[], pred=[]),
        ]
        out = correctly_empty_rate(inst)
        assert out["empty_gold_n"] == 2
        assert out["correctly_empty_n"] == 2
        assert out["rate"] == 1.0
        assert out["avg_fp_per_empty"] == 0.0

    def test_all_spurious(self):
        # Empty gold but model predicted something every time
        inst = [
            Instance(text="x" * 10, gold=[], pred=[_span("PERSON", 0, 3)]),
            Instance(text="x" * 10, gold=[], pred=[_span("EMAIL", 0, 3), _span("PHONE", 4, 7)]),
        ]
        out = correctly_empty_rate(inst)
        assert out["empty_gold_n"] == 2
        assert out["correctly_empty_n"] == 0
        assert out["rate"] == 0.0
        # 1 + 2 FPs across 2 empty-gold instances ⇒ 1.5 avg
        assert out["avg_fp_per_empty"] == 1.5

    def test_mixed(self):
        inst = [
            Instance(text="x" * 10, gold=[], pred=[]),  # correct
            Instance(text="x" * 10, gold=[], pred=[_span("PERSON", 0, 3)]),  # spurious
            Instance(text="x" * 10, gold=[_span("PERSON", 0, 3)], pred=[]),  # not empty-gold
        ]
        out = correctly_empty_rate(inst)
        assert out["empty_gold_n"] == 2  # 3rd doesn't count
        assert out["correctly_empty_n"] == 1
        assert out["rate"] == 0.5
        assert out["avg_fp_per_empty"] == 0.5  # 1 FP across 2 empty-gold instances
