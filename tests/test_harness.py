"""Tests for eval/harness.py — uses synthetic result JSONs in tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness import (
    INJECT_END,
    INJECT_START,
    discover_predictors,
    inject_into_file,
    load_predictor_results,
    render_markdown,
)


def _make_main_result(f1: float, schema: float = 1.0, halluc: float = 0.0,
                      p50_s: float = 0.020, self_hosted: bool = True) -> dict:
    return {
        "predictor": "fake",
        "n_examples": 100,
        "strict_f1": {
            "overall": {"tp": int(50 * f1), "fp": 5, "fn": 5,
                        "precision": f1, "recall": f1, "f1": f1},
            "per_type": {
                "PERSON": {"tp": 10, "fp": 2, "fn": 3, "precision": 0.83, "recall": 0.77, "f1": 0.80},
                "EMAIL": {"tp": 20, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
                # Other types: zero counts
                "PHONE": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                "ADDRESS": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                "DOB": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                "GOV_ID": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                "FINANCIAL": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                "MEDICAL_ID": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                "IP_ADDRESS": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            },
        },
        "partial_f1": {"overall": {"f1": f1 * 1.05, "tp": 0, "fp": 0, "fn": 0,
                                   "precision": 0, "recall": 0}},
        "schema_validity": {"rate": schema, "valid": int(100 * schema), "total": 100, "n_failed": 0, "failures": []},
        "hallucination": {"rate": halluc, "hallucinated": 0, "total_predictions": 100, "examples": []},
        "correctly_empty": {"empty_gold_n": 20, "correctly_empty_n": 15, "rate": 0.75, "avg_fp_per_empty": 0.3},
        "perf": {
            "latency": {"n": 100, "p50_s": p50_s, "p99_s": p50_s * 3, "mean_s": p50_s * 1.2,
                        "min_s": p50_s * 0.5, "max_s": p50_s * 4},
            "cost": {"self_hosted": self_hosted, "n_calls": 100} if self_hosted else {
                "self_hosted": False, "n_calls": 100, "n_with_token_counts": 100,
                "total_in_tokens": 50000, "total_out_tokens": 5000,
                "input_per_1m_usd": 0.15, "output_per_1m_usd": 0.6,
                "total_usd": 0.0105, "per_call_usd": 0.000105, "per_1k_usd": 0.105,
            },
        },
        "errors": [],
    }


def _make_adversarial_result(f1: float = 0.5) -> dict:
    return {
        "predictor": "fake",
        "n_examples": 50,
        "strict_f1": {"overall": {"tp": 30, "fp": 10, "fn": 10, "precision": 0.75, "recall": 0.75, "f1": f1},
                      "per_type": {}},
        "partial_f1": {"overall": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": f1 + 0.1}},
        "schema_validity": {"rate": 1.0, "valid": 50, "total": 50, "n_failed": 0, "failures": []},
        "hallucination": {"rate": 0.0, "hallucinated": 0, "total_predictions": 60, "examples": []},
        "correctly_empty": {"empty_gold_n": 10, "correctly_empty_n": 7, "rate": 0.7, "avg_fp_per_empty": 0.5},
        "perf": {"latency": {"p50_s": 0.005, "p99_s": 0.05, "mean_s": 0.01, "min_s": 0.001, "max_s": 0.1, "n": 50},
                 "cost": {"self_hosted": True, "n_calls": 50}},
        "errors": [],
        "by_category": {
            "phone_format": {"n": 4, "strict_f1": {"tp": 3, "fp": 1, "fn": 1, "precision": 0.75, "recall": 0.75, "f1": 0.75},
                             "partial_f1": {"f1": 0.8}, "correctly_empty": {"empty_gold_n": 0, "rate": 0.0, "correctly_empty_n": 0, "avg_fp_per_empty": 0}},
            "reference_no_value": {"n": 3, "strict_f1": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
                                   "partial_f1": {"f1": 0}, "correctly_empty": {"empty_gold_n": 3, "correctly_empty_n": 3, "rate": 1.0, "avg_fp_per_empty": 0}},
        },
    }


def _make_heldout_result(recall: float = 0.6) -> dict:
    tp = int(100 * recall)
    return {
        "predictor": "fake",
        "n_examples": 50,
        "strict_f1": {"overall": {"tp": tp, "fp": 200, "fn": 100 - tp, "precision": tp / (tp + 200), "recall": recall, "f1": 0.2},
                      "per_type": {}},
        "partial_f1": {"overall": {"f1": 0.25, "tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0}},
        "schema_validity": {"rate": 1.0, "valid": 50, "total": 50, "n_failed": 0, "failures": []},
        "hallucination": {"rate": 0.0, "hallucinated": 0, "total_predictions": 0, "examples": []},
        "correctly_empty": {"empty_gold_n": 0, "correctly_empty_n": 0, "rate": 0.0, "avg_fp_per_empty": 0.0},
        "perf": {"latency": {"p50_s": 0.03, "p99_s": 0.1, "mean_s": 0.04, "min_s": 0.01, "max_s": 0.2, "n": 50},
                 "cost": {"self_hosted": True, "n_calls": 50}},
        "errors": [],
    }


def _setup_results(results_dir: Path, predictor: str, **kwargs) -> None:
    pdir = results_dir / predictor
    pdir.mkdir(parents=True, exist_ok=True)
    if "main" in kwargs:
        (pdir / "main.json").write_text(json.dumps(kwargs["main"]))
    if "heldout" in kwargs:
        (pdir / "heldout.json").write_text(json.dumps(kwargs["heldout"]))
    if "adversarial" in kwargs:
        (pdir / "adversarial.json").write_text(json.dumps(kwargs["adversarial"]))


# --- discover_predictors ---


class TestDiscoverPredictors:
    def test_empty_dir(self, tmp_path: Path):
        assert discover_predictors(tmp_path) == []

    def test_missing_dir(self, tmp_path: Path):
        assert discover_predictors(tmp_path / "nope") == []

    def test_finds_predictors_with_json(self, tmp_path: Path):
        _setup_results(tmp_path, "presidio", main=_make_main_result(0.5))
        _setup_results(tmp_path, "openai", main=_make_main_result(0.7))
        assert discover_predictors(tmp_path) == ["openai", "presidio"]

    def test_skips_calibration_dir(self, tmp_path: Path):
        _setup_results(tmp_path, "presidio", main=_make_main_result(0.5))
        (tmp_path / "calibration").mkdir()
        (tmp_path / "calibration" / "x.json").write_text("{}")
        assert discover_predictors(tmp_path) == ["presidio"]

    def test_skips_empty_predictor_dir(self, tmp_path: Path):
        (tmp_path / "presidio").mkdir()
        # No JSON files inside
        assert discover_predictors(tmp_path) == []


# --- load_predictor_results ---


class TestLoadResults:
    def test_loads_main(self, tmp_path: Path):
        _setup_results(tmp_path, "presidio", main=_make_main_result(0.5))
        out = load_predictor_results(tmp_path, "presidio")
        assert "main" in out
        assert out["main"]["strict_f1"]["overall"]["f1"] == 0.5

    def test_loads_calibration_when_present(self, tmp_path: Path):
        _setup_results(tmp_path, "presidio", main=_make_main_result(0.5))
        cal_dir = tmp_path / "calibration" / "presidio"
        cal_dir.mkdir(parents=True)
        (cal_dir / "agreement.json").write_text(
            json.dumps({"correlation": {"r": 0.75}, "passes_target": True,
                       "judge_a": "openai:gpt-4o", "judge_b": "gemini:gemini-2.5-pro",
                       "n_aligned": 50})
        )
        out = load_predictor_results(tmp_path, "presidio")
        assert out["calibration"]["correlation"]["r"] == 0.75


# --- render_markdown ---


class TestRenderMarkdown:
    def test_empty_data_message(self):
        md = render_markdown({})
        assert "No predictor results" in md

    def test_basic_render(self):
        md = render_markdown({
            "presidio": {
                "main": _make_main_result(0.438),
                "adversarial": _make_adversarial_result(0.487),
                "heldout": _make_heldout_result(0.63),
            }
        })
        # Smoke checks
        assert "Headline (main test)" in md
        assert "presidio" in md
        assert "0.438" in md
        assert "EMAIL" in md  # per-type section
        assert "Adversarial" in md
        assert "Held-out" in md
        # Generated-by comment
        assert "generated by eval.harness" in md

    def test_multi_predictor_table(self):
        md = render_markdown({
            "presidio": {"main": _make_main_result(0.4)},
            "openai": {"main": _make_main_result(0.7, self_hosted=False)},
        })
        # Both names should appear in the per-type header
        assert "presidio" in md
        assert "openai" in md
        # OpenAI should show a $ cost
        assert "$" in md

    def test_adversarial_per_category_shows_empty_for_negative_controls(self):
        md = render_markdown({
            "presidio": {"adversarial": _make_adversarial_result(0.5)},
        })
        # reference_no_value has all empty gold; should render "empty 100%"
        assert "empty 100%" in md

    def test_calibration_section_only_when_present(self):
        md_no = render_markdown({"presidio": {"main": _make_main_result(0.4)}})
        assert "Calibration" not in md_no

        md_yes = render_markdown({
            "presidio": {
                "main": _make_main_result(0.4),
                "calibration": {
                    "judge_a": "openai:gpt-4o", "judge_b": "gemini:gemini-2.5-pro",
                    "n_aligned": 48, "passes_target": True,
                    "correlation": {"r": 0.74, "p_value": 0.001, "n": 48},
                },
            }
        })
        assert "Calibration" in md_yes
        assert "0.740" in md_yes


# --- inject_into_file ---


class TestInject:
    def test_inject_replaces_between_markers(self, tmp_path: Path):
        target = tmp_path / "README.md"
        target.write_text(
            f"intro\n\n{INJECT_START}\nold content\n{INJECT_END}\n\noutro",
            encoding="utf-8",
        )
        ok = inject_into_file(target, "## new content\n")
        assert ok
        new = target.read_text(encoding="utf-8")
        assert "new content" in new
        assert "old content" not in new
        assert "intro" in new
        assert "outro" in new

    def test_inject_idempotent(self, tmp_path: Path):
        target = tmp_path / "README.md"
        target.write_text(f"{INJECT_START}\n{INJECT_END}", encoding="utf-8")
        inject_into_file(target, "first\n")
        inject_into_file(target, "second\n")
        new = target.read_text(encoding="utf-8")
        assert "second" in new
        assert "first" not in new

    def test_inject_missing_markers_fails(self, tmp_path: Path):
        target = tmp_path / "README.md"
        target.write_text("no markers here", encoding="utf-8")
        assert inject_into_file(target, "anything") is False

    def test_inject_missing_file(self, tmp_path: Path):
        assert inject_into_file(tmp_path / "nope.md", "x") is False
