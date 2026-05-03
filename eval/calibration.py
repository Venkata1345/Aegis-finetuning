"""Cross-LLM-judge calibration for the eval harness.

Per the project decision recorded on 2026-05-03: NO human gold standard.
Two independent LLM judges (GPT-4o + Gemini 2.5 Pro by default) score the
same N predictor outputs against the gold spans on a 1-5 rubric. We
report the Pearson r between the judges' score vectors as the reliability
proxy and document the limitation honestly in the README.

If Pearson r < 0.6, the rubric isn't sharp enough — iterate JUDGE_RUBRIC
and re-run before any judge-derived number is trusted.

Layout:

  Inputs (read):
    eval/results/<predictor>/adversarial.log.jsonl   per-instance log

  Outputs (written):
    eval/results/calibration/<predictor>/scores.jsonl    one row per (case, judge)
    eval/results/calibration/<predictor>/agreement.json  Pearson r + summary

The judge models are deliberately STRONGER than the baselines being scored
(GPT-4o not GPT-4o-mini; Gemini 2.5 Pro not Flash) — judges should be at
least as capable as the predictors they evaluate.

Usage:
    python -m eval.calibration --predictor presidio --sample 50

Requires OPENAI_API_KEY and GOOGLE_API_KEY env vars at run time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CALIB_DIR = RESULTS_DIR / "calibration"

DEFAULT_SAMPLE_N = 50
DEFAULT_SEED = 42
PEARSON_TARGET = 0.6  # below this, the rubric must be iterated


JUDGE_RUBRIC = """You are evaluating the output of a PII detection model.

For one input text, you see:
  - GOLD: the curated correct PII spans (ground truth)
  - PRED: the spans the model produced

Rate how correct PRED is on a strict 1-5 integer scale:

  5 - Excellent. Every gold span is captured with the right type and
      offsets. No spurious spans. Boundaries match exactly.
  4 - Good. Every gold type is captured. Boundaries may be slightly off
      (1-2 chars). At most 1 minor spurious span.
  3 - Mixed. About half of gold spans captured OR full gold coverage but
      2-3 spurious spans OR several boundary or type errors.
  2 - Poor. Most gold spans missed OR many spurious spans (4+) OR
      systematic type confusion.
  1 - Bad. The prediction is essentially wrong: little/no overlap with
      gold, wrong types throughout, or empty when gold is non-empty.

Type correctness matters: flagging an EMAIL as PHONE is wrong, even with
matching offsets. Empty PRED on empty GOLD = score 5 (perfectly correct).

INPUT TEXT (truncated if long):
{text}

GOLD SPANS:
{gold_json}

PREDICTED SPANS:
{pred_json}

Reply with ONLY a single integer: 1, 2, 3, 4, or 5. No words, no punctuation,
no explanation."""

MAX_TEXT_CHARS_IN_PROMPT = 2000


# --- judges ----------------------------------------------------------------


@dataclass
class JudgeScore:
    case_id: str
    judge_name: str
    score: int | None
    raw_response: str
    error: str | None
    latency_s: float


def parse_score(raw: str) -> int | None:
    """Extract the first 1-5 integer from a judge response. None if absent."""
    if not raw:
        return None
    m = re.search(r"[1-5]", raw)
    return int(m.group(0)) if m else None


def _format_prompt(text: str, gold: list[dict], pred: list[dict]) -> str:
    truncated = text if len(text) <= MAX_TEXT_CHARS_IN_PROMPT else (
        text[:MAX_TEXT_CHARS_IN_PROMPT] + "  […truncated…]"
    )
    return JUDGE_RUBRIC.format(
        text=truncated,
        gold_json=json.dumps(gold, ensure_ascii=False),
        pred_json=json.dumps(pred, ensure_ascii=False),
    )


class Judge(ABC):
    name: str

    @abstractmethod
    def _ask(self, prompt: str) -> str:
        """Send the prompt and return the raw text response."""

    def score(self, case_id: str, text: str, gold: list[dict], pred: list[dict]) -> JudgeScore:
        prompt = _format_prompt(text, gold, pred)
        start = time.perf_counter()
        try:
            raw = self._ask(prompt)
        except Exception as e:  # noqa: BLE001
            return JudgeScore(
                case_id=case_id,
                judge_name=self.name,
                score=None,
                raw_response="",
                error=f"{type(e).__name__}: {e}",
                latency_s=time.perf_counter() - start,
            )
        latency = time.perf_counter() - start
        score = parse_score(raw)
        return JudgeScore(
            case_id=case_id,
            judge_name=self.name,
            score=score,
            raw_response=raw,
            error=None if score is not None else "unparseable_response",
            latency_s=latency,
        )


class OpenAIJudge(Judge):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot run OpenAI judge.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def _ask(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        return resp.choices[0].message.content or ""


class GeminiJudge(Judge):
    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set; cannot run Gemini judge.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model_obj = genai.GenerativeModel(model_name=model)
        self.name = f"gemini:{model}"

    def _ask(self, prompt: str) -> str:
        resp = self.model_obj.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 8},
        )
        return resp.text if hasattr(resp, "text") else ""


# --- correlation -----------------------------------------------------------


def compute_pearson(scores_a: list[int], scores_b: list[int]) -> dict:
    """Pearson r between two same-length integer score lists.

    Returns {r, p_value, n, note?}. n<3 returns r=None with a note.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Mismatched lengths: {len(scores_a)} vs {len(scores_b)}")
    n = len(scores_a)
    if n < 3:
        return {"r": None, "p_value": None, "n": n, "note": "n<3 — Pearson requires more samples"}
    # Need variance in both; if either is constant, r is undefined.
    if len(set(scores_a)) < 2 or len(set(scores_b)) < 2:
        return {"r": None, "p_value": None, "n": n, "note": "one judge gave a constant score"}

    from scipy.stats import pearsonr  # local import — scipy is heavy

    r, p = pearsonr(scores_a, scores_b)
    return {"r": float(r), "p_value": float(p), "n": n}


# --- runner ---------------------------------------------------------------


@dataclass
class CalibrationResult:
    predictor: str
    scores: list[JudgeScore] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _load_log(log_path: Path) -> list[dict]:
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_cases(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def run_calibration(
    predictor: str,
    judges: list[Judge],
    log_path: Path | None = None,
    sample_n: int = DEFAULT_SAMPLE_N,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
) -> CalibrationResult:
    """Run cross-judge calibration for one predictor.

    Args:
        predictor: predictor name (matches eval/results/<predictor>/ dir).
        judges: list of two Judge instances (typically OpenAIJudge + GeminiJudge).
        log_path: per-instance log JSONL. Default: adversarial.log.jsonl.
        sample_n: how many cases to sample.
        seed: RNG seed for sample selection.
        out_dir: where to write outputs. Default: eval/results/calibration/<predictor>/.

    Returns CalibrationResult with all per-case scores and the summary.
    """
    if len(judges) != 2:
        raise ValueError("Cross-judge calibration requires exactly 2 judges.")

    log_path = log_path or RESULTS_DIR / predictor / "adversarial.log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Predictor log not found: {log_path}")

    out_dir = out_dir or CALIB_DIR / predictor
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_log(log_path)
    sample = _sample_cases(rows, sample_n, seed)
    print(f"Sampled {len(sample)} of {len(rows)} cases (seed={seed})")

    all_scores: list[JudgeScore] = []
    for i, case in enumerate(sample, 1):
        case_id = case.get("id", f"case_{i}")
        text = case.get("raw") or ""  # log records the raw model output as 'raw'
        # Reconstruct text from the log: log doesn't store the original input
        # text directly; we must trust 'text_len' was set and pull from the
        # source dataset if needed. For now, the judge prompt uses gold/pred
        # which is sufficient when the input text isn't critical (the judge
        # is rating PII coverage, not text comprehension).
        # If the log includes the source text, use it; otherwise empty.
        input_text = case.get("text", text)
        gold = case.get("gold") or []
        pred = case.get("pred") or []
        for j in judges:
            print(f"  [{i}/{len(sample)}] {case_id} <- {j.name}", end="", flush=True)
            score = j.score(case_id, input_text, gold, pred)
            tag = f"{score.score}" if score.score is not None else f"ERR({score.error})"
            print(f"  {tag}")
            all_scores.append(score)

    # Write per-case JSONL
    scores_path = out_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as f:
        for s in all_scores:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

    # Build score vectors aligned by case_id, keeping only cases where BOTH
    # judges produced a parseable score
    by_case: dict[str, dict[str, int]] = {}
    for s in all_scores:
        if s.score is None:
            continue
        by_case.setdefault(s.case_id, {})[s.judge_name] = s.score

    aligned = [
        (cid, scores) for cid, scores in by_case.items()
        if all(j.name in scores for j in judges)
    ]
    a_vec = [scores[judges[0].name] for _, scores in aligned]
    b_vec = [scores[judges[1].name] for _, scores in aligned]

    correlation = compute_pearson(a_vec, b_vec)
    summary = {
        "predictor": predictor,
        "judge_a": judges[0].name,
        "judge_b": judges[1].name,
        "n_sampled": len(sample),
        "n_aligned": len(aligned),
        "correlation": correlation,
        "pearson_target": PEARSON_TARGET,
        "passes_target": (
            correlation.get("r") is not None and correlation["r"] >= PEARSON_TARGET
        ),
        "notes": (
            "Cross-LLM-judge calibration. No human gold standard. "
            f"Target r >= {PEARSON_TARGET}. If failing, iterate JUDGE_RUBRIC "
            "in eval/calibration.py and re-run."
        ),
    }
    summary_path = out_dir / "agreement.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return CalibrationResult(predictor=predictor, scores=all_scores, summary=summary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True, help="Predictor name (e.g. presidio).")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--openai-model", default="gpt-4o", help="Judge model for OpenAI side."
    )
    ap.add_argument(
        "--gemini-model", default="gemini-2.5-pro", help="Judge model for Gemini side."
    )
    args = ap.parse_args()

    judges = [
        OpenAIJudge(model=args.openai_model),
        GeminiJudge(model=args.gemini_model),
    ]
    result = run_calibration(
        predictor=args.predictor,
        judges=judges,
        sample_n=args.sample,
        seed=args.seed,
    )
    s = result.summary
    print()
    print(f"Calibration: {s['judge_a']} vs {s['judge_b']} on {s['predictor']}")
    print(f"  n aligned = {s['n_aligned']}/{s['n_sampled']}")
    corr = s["correlation"]
    if corr.get("r") is None:
        print(f"  r = N/A  ({corr.get('note', '')})")
    else:
        print(f"  Pearson r = {corr['r']:.3f}  (p={corr['p_value']:.3g}, n={corr['n']})")
    print(f"  passes target (r >= {s['pearson_target']}): {s['passes_target']}")
    if not s["passes_target"]:
        print("  ACTION: iterate JUDGE_RUBRIC in eval/calibration.py and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
