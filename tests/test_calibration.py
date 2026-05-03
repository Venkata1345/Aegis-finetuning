"""Tests for eval/calibration.py — uses mock judges (no API).

Coverage:
  - parse_score: integer in middle, no integer, out of range, multiple
  - compute_pearson: identical, opposite, n<3, mismatched length, constant
  - run_calibration end-to-end with mock judges over a tiny synthetic log
  - sampling determinism
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.calibration import (
    Judge,
    JudgeScore,
    _sample_cases,
    compute_pearson,
    parse_score,
    run_calibration,
)


# --- parse_score -----------------------------------------------------------


class TestParseScore:
    def test_bare_integer(self):
        assert parse_score("4") == 4

    def test_integer_with_text(self):
        assert parse_score("The score is 3 because...") == 3

    def test_no_integer(self):
        assert parse_score("I cannot rate this") is None

    def test_empty(self):
        assert parse_score("") is None

    def test_first_in_range_wins(self):
        # Out-of-range numbers ignored; first valid 1-5 wins
        assert parse_score("0 is wrong, the answer is 4") == 4

    def test_zero_ignored(self):
        # 0 is not valid in our 1-5 scale
        assert parse_score("0") is None

    def test_six_ignored(self):
        # 6 falls outside scale
        assert parse_score("6") is None

    def test_multiple_picks_first(self):
        assert parse_score("4 5 3 2") == 4


# --- compute_pearson -------------------------------------------------------


class TestComputePearson:
    def test_identical(self):
        out = compute_pearson([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert out["r"] == 1.0
        assert out["n"] == 5

    def test_perfect_negative(self):
        out = compute_pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert abs(out["r"] - (-1.0)) < 1e-9

    def test_n_less_than_3(self):
        out = compute_pearson([1, 2], [2, 3])
        assert out["r"] is None
        assert "n<3" in out.get("note", "")

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            compute_pearson([1, 2, 3], [1, 2])

    def test_constant_returns_none(self):
        # If one judge gave the same score every time, Pearson is undefined
        out = compute_pearson([3, 3, 3, 3], [1, 2, 3, 4])
        assert out["r"] is None
        assert "constant" in out.get("note", "").lower()


# --- _sample_cases ---------------------------------------------------------


class TestSampleCases:
    def test_deterministic_with_seed(self):
        rows = [{"id": str(i)} for i in range(20)]
        a = _sample_cases(rows, 5, seed=42)
        b = _sample_cases(rows, 5, seed=42)
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_seed_changes_sample(self):
        rows = [{"id": str(i)} for i in range(20)]
        a = _sample_cases(rows, 5, seed=1)
        b = _sample_cases(rows, 5, seed=2)
        # With 20 items and 5 samples, two different seeds should differ
        assert [r["id"] for r in a] != [r["id"] for r in b]

    def test_sample_larger_than_population(self):
        rows = [{"id": "x"}, {"id": "y"}]
        out = _sample_cases(rows, 10, seed=0)
        assert len(out) == 2


# --- run_calibration with mock judges --------------------------------------


class _MockJudge(Judge):
    """Returns canned scores from a list, in order. None ⇒ unparseable."""

    def __init__(self, name: str, scores: list[int | None]) -> None:
        self.name = name
        self._scores = list(scores)
        self._idx = 0

    def _ask(self, prompt: str) -> str:
        if self._idx >= len(self._scores):
            return ""
        s = self._scores[self._idx]
        self._idx += 1
        return "" if s is None else str(s)


def _write_log(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "id": f"case_{i}",
                "text": f"text-{i}",
                "gold": [],
                "pred": [],
                "raw": "[]",
            }
            f.write(json.dumps(row) + "\n")


class TestRunCalibration:
    def test_perfect_agreement(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        _write_log(log, 5)
        j1 = _MockJudge("mockA", [3, 3, 3, 3, 3])
        j2 = _MockJudge("mockB", [3, 3, 3, 3, 3])
        # constant scores ⇒ correlation undefined
        result = run_calibration(
            predictor="testpred",
            judges=[j1, j2],
            log_path=log,
            sample_n=5,
            out_dir=tmp_path / "out",
        )
        assert result.summary["n_aligned"] == 5
        assert result.summary["correlation"]["r"] is None
        assert result.summary["passes_target"] is False

    def test_strong_correlation(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        _write_log(log, 5)
        j1 = _MockJudge("mockA", [1, 2, 3, 4, 5])
        j2 = _MockJudge("mockB", [1, 2, 3, 4, 5])  # identical
        result = run_calibration(
            predictor="testpred",
            judges=[j1, j2],
            log_path=log,
            sample_n=5,
            out_dir=tmp_path / "out",
        )
        assert result.summary["correlation"]["r"] == 1.0
        assert result.summary["passes_target"] is True

    def test_weak_correlation_fails_target(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        _write_log(log, 6)
        j1 = _MockJudge("mockA", [1, 5, 2, 5, 1, 5])
        j2 = _MockJudge("mockB", [5, 1, 5, 1, 5, 1])  # opposite
        result = run_calibration(
            predictor="testpred",
            judges=[j1, j2],
            log_path=log,
            sample_n=6,
            out_dir=tmp_path / "out",
        )
        # Strongly negative correlation — fails the 0.6 target
        assert result.summary["passes_target"] is False
        assert result.summary["correlation"]["r"] < 0

    def test_writes_outputs(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        _write_log(log, 3)
        j1 = _MockJudge("mockA", [1, 2, 3])
        j2 = _MockJudge("mockB", [1, 2, 3])
        out_dir = tmp_path / "out"
        run_calibration(
            predictor="testpred", judges=[j1, j2], log_path=log,
            sample_n=3, out_dir=out_dir,
        )
        scores_path = out_dir / "scores.jsonl"
        agreement_path = out_dir / "agreement.json"
        assert scores_path.exists()
        assert agreement_path.exists()
        # 3 cases × 2 judges = 6 score rows
        rows = scores_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(rows) == 6

    def test_unparseable_judge_response_excluded(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        _write_log(log, 4)
        j1 = _MockJudge("mockA", [1, 2, 3, 4])
        j2 = _MockJudge("mockB", [None, 2, 3, 4])  # first response unparseable
        result = run_calibration(
            predictor="testpred",
            judges=[j1, j2],
            log_path=log,
            sample_n=4,
            out_dir=tmp_path / "out",
        )
        # Only 3 cases have both judges scoring
        assert result.summary["n_aligned"] == 3
        assert result.summary["correlation"]["n"] == 3

    def test_requires_exactly_two_judges(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        _write_log(log, 3)
        with pytest.raises(ValueError):
            run_calibration(
                predictor="x",
                judges=[_MockJudge("a", [3])],
                log_path=log,
                sample_n=3,
                out_dir=tmp_path / "out",
            )

    def test_missing_log_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            run_calibration(
                predictor="x",
                judges=[_MockJudge("a", []), _MockJudge("b", [])],
                log_path=tmp_path / "nope.jsonl",
                sample_n=1,
                out_dir=tmp_path / "out",
            )


# --- Judge.score wraps exceptions -------------------------------------------


class _BoomJudge(Judge):
    name = "boom"

    def _ask(self, prompt: str) -> str:
        raise RuntimeError("kaboom")


class TestJudgeErrorHandling:
    def test_exception_caught_returns_jscore_with_error(self):
        j = _BoomJudge()
        s = j.score("c1", "text", [], [])
        assert s.score is None
        assert s.error is not None
        assert "kaboom" in s.error
