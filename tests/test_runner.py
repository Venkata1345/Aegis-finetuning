"""Tests for eval/runner.py — uses a deterministic FakePredictor (no network).

Coverage targets the integration glue: dataset loading, per-instance logging,
metric aggregation, by_category grouping, and graceful handling of
schema-invalid model output.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.base import Predictor, Pricing
from eval.runner import evaluate


class FakePredictor(Predictor):
    """Returns a fixed raw response per text. None ⇒ "[]" ⇒ no spans."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        name: str = "fake",
        in_tok: int | None = None,
        out_tok: int | None = None,
    ) -> None:
        super().__init__(name=name, pricing=Pricing())
        self.responses = responses or {}
        self._in_tok = in_tok
        self._out_tok = out_tok

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        return self.responses.get(text, "[]"), self._in_tok, self._out_tok


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestRunner:
    def test_perfect_predictor(self, tmp_path: Path):
        rows = [{
            "id": "r1",
            "text": "Alice",
            "spans": [{"type": "PERSON", "start": 0, "end": 5, "text": "Alice"}],
        }]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)
        responses = {"Alice": '[{"type":"PERSON","start":0,"end":5,"text":"Alice"}]'}
        result = evaluate(FakePredictor(responses), ds)
        assert result["strict_f1"]["overall"]["f1"] == 1.0
        assert result["schema_validity"]["rate"] == 1.0
        assert result["hallucination"]["rate"] == 0.0
        assert result["n_examples"] == 1

    def test_invalid_json_caught(self, tmp_path: Path):
        rows = [{"text": "hello", "spans": []}]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)
        result = evaluate(FakePredictor({"hello": "not json{"}), ds)
        assert result["schema_validity"]["rate"] == 0.0
        assert result["strict_f1"]["overall"]["tp"] == 0

    def test_log_path_writes_records(self, tmp_path: Path):
        rows = [{"id": "x", "text": "hi", "spans": []}]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)
        log = tmp_path / "log.jsonl"
        evaluate(FakePredictor({"hi": "[]"}), ds, log_path=log)
        assert log.exists()
        records = log.read_text(encoding="utf-8").strip().split("\n")
        assert len(records) == 1
        rec = json.loads(records[0])
        assert rec["id"] == "x"
        assert rec["schema_valid"] is True
        assert rec["error"] is None

    def test_by_category_groups(self, tmp_path: Path):
        rows = [
            {"id": "1", "category": "phone_format", "text": "a", "spans": []},
            {"id": "2", "category": "phone_format", "text": "b", "spans": []},
            {"id": "3", "category": "near_miss",   "text": "c", "spans": []},
        ]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)
        result = evaluate(FakePredictor({}), ds, by_category=True)
        assert "phone_format" in result["by_category"]
        assert result["by_category"]["phone_format"]["n"] == 2
        assert result["by_category"]["near_miss"]["n"] == 1

    def test_max_examples(self, tmp_path: Path):
        rows = [{"id": str(i), "text": f"t{i}", "spans": []} for i in range(10)]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)
        result = evaluate(FakePredictor({}), ds, max_examples=3)
        assert result["n_examples"] == 3

    def test_self_hosted_cost_summary(self, tmp_path: Path):
        rows = [{"text": "a", "spans": []}]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)
        result = evaluate(FakePredictor({}), ds)
        assert result["perf"]["cost"]["self_hosted"] is True

    def test_priced_cost_summary(self, tmp_path: Path):
        rows = [{"text": "a", "spans": []}, {"text": "b", "spans": []}]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)

        class PricedFake(FakePredictor):
            def __init__(self) -> None:
                super().__init__(in_tok=1000, out_tok=200)
                self.pricing = Pricing(input_per_1m=1.0, output_per_1m=2.0)

        result = evaluate(PricedFake(), ds)
        cost = result["perf"]["cost"]
        assert cost["self_hosted"] is False
        # 2 calls × (1000 × $1 + 200 × $2) / 1e6 = 2 × $0.0014 = $0.0028
        assert abs(cost["total_usd"] - 0.0028) < 1e-9

    def test_predictor_exception_recorded(self, tmp_path: Path):
        rows = [{"text": "boom", "spans": []}]
        ds = tmp_path / "data.jsonl"
        _write_jsonl(ds, rows)

        class Boom(Predictor):
            def __init__(self) -> None:
                super().__init__(name="boom", pricing=Pricing())

            def _predict_impl(self, text):
                raise RuntimeError("kaboom")

        result = evaluate(Boom(), ds)
        assert len(result["errors"]) == 1
        assert "kaboom" in result["errors"][0]
        assert result["schema_validity"]["rate"] == 0.0


class TestStripCodeFences:
    """Imported here to keep predictor-related tests close together."""

    def test_no_fences_passthrough(self):
        from baselines.base import strip_code_fences

        assert strip_code_fences("[]") == "[]"
        assert strip_code_fences('  [{"type":"PERSON"}]  ') == '[{"type":"PERSON"}]'

    def test_json_fence(self):
        from baselines.base import strip_code_fences

        s = "```json\n[]\n```"
        assert strip_code_fences(s) == "[]"

    def test_plain_fence(self):
        from baselines.base import strip_code_fences

        s = "```\n[1,2,3]\n```"
        assert strip_code_fences(s) == "[1,2,3]"

    def test_fence_with_trailing_blank(self):
        from baselines.base import strip_code_fences

        s = "```json\n[]\n\n```\n"
        assert strip_code_fences(s) == "[]"
