"""Run a single baseline against the standard eval suite.

Targets three datasets:
  - main test (data/processed/validation.jsonl)  — 7,946 rows
  - held-out (data/eval/heldout.jsonl)           — 50 rows
  - adversarial (data/eval/adversarial.jsonl)    — 50 rows, by_category=True

Writes results to eval/results/<predictor>/<dataset>.json (high-level metrics)
and eval/results/<predictor>/<dataset>.log.jsonl (per-instance for error
analysis). Compact summary printed to stdout for human inspection.

Run multiple times with different predictors; the harness (4i) reads these
JSONs to assemble the comparison table.

Usage:
    python -m eval.run_baseline presidio              # full run
    python -m eval.run_baseline presidio --smoke      # heldout + adversarial only
    python -m eval.run_baseline openai --max-main 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from baselines.base import Predictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"


def _build_predictor(name: str) -> Predictor:
    """Lazy import — only the requested predictor's deps need to be reachable."""
    if name == "presidio":
        from baselines.presidio_caller import PresidioPredictor

        return PresidioPredictor()
    if name == "openai":
        from baselines.openai_caller import OpenAIPredictor

        return OpenAIPredictor()
    if name == "gemini":
        from baselines.gemini_caller import GeminiPredictor

        return GeminiPredictor()
    if name == "qwen-base":
        from baselines.qwen_base import BaseQwenPredictor

        return BaseQwenPredictor()
    if name == "aegis":
        from baselines.aegis_caller import AegisPredictor

        return AegisPredictor()
    raise ValueError(f"Unknown predictor: {name}")


def _print_compact(name: str, result: dict) -> None:
    sf = result["strict_f1"]["overall"]
    pf = result["partial_f1"]["overall"]
    sv = result["schema_validity"]
    h = result["hallucination"]
    ce = result["correctly_empty"]
    lat = result["perf"]["latency"]
    cost = result["perf"]["cost"]
    print(f"  {name}: n={result['n_examples']}")
    print(
        f"    strict F1 = {sf['f1']:.3f}  "
        f"(P={sf['precision']:.3f} R={sf['recall']:.3f}; "
        f"TP/FP/FN = {sf['tp']}/{sf['fp']}/{sf['fn']})"
    )
    print(f"    partial F1 = {pf['f1']:.3f}")
    print(f"    schema valid = {sv['rate']:.1%}  ({sv['valid']}/{sv['total']})")
    print(f"    hallucination = {h['rate']:.1%}  ({h['hallucinated']}/{h['total_predictions']})")
    if ce["empty_gold_n"] > 0:
        print(
            f"    correctly empty = {ce['rate']:.1%}  "
            f"({ce['correctly_empty_n']}/{ce['empty_gold_n']})  "
            f"avg_fp_per_empty={ce['avg_fp_per_empty']:.2f}"
        )
    print(
        f"    latency  p50={lat['p50_s'] * 1000:.0f}ms  "
        f"p99={lat['p99_s'] * 1000:.0f}ms  "
        f"mean={lat['mean_s'] * 1000:.0f}ms"
    )
    if cost.get("self_hosted"):
        print(f"    cost: self-hosted")
    else:
        print(f"    cost: ${cost['total_usd']:.4f} total, ${cost['per_1k_usd']:.4f}/1k calls")
    if result.get("errors"):
        print(f"    errors: {len(result['errors'])} (first: {result['errors'][0][:120]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "predictor",
        choices=["presidio", "openai", "gemini", "qwen-base", "aegis"],
        help="Which baseline to run.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Skip the main test set; run only heldout + adversarial.",
    )
    ap.add_argument(
        "--max-main",
        type=int,
        default=None,
        help="Cap rows from the main test set (useful for slow predictors).",
    )
    args = ap.parse_args()

    # (path, by_category, max_examples)
    datasets: dict[str, tuple[Path, bool, int | None]] = {
        "heldout": (DATA_EVAL / "heldout.jsonl", False, None),
        "adversarial": (DATA_EVAL / "adversarial.jsonl", True, None),
    }
    if not args.smoke:
        datasets["main"] = (DATA_PROCESSED / "validation.jsonl", False, args.max_main)

    print(f"Building predictor: {args.predictor}")
    predictor = _build_predictor(args.predictor)
    print(f"  -> {predictor.name}")

    from eval.runner import evaluate

    out_dir = RESULTS_DIR / args.predictor
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting results to {out_dir}\n")

    for ds_name, (path, by_cat, max_n) in datasets.items():
        if not path.exists():
            print(f"  skip {ds_name}: missing {path}", file=sys.stderr)
            continue
        print(f"Running {predictor.name} on {ds_name}...")
        result = evaluate(
            predictor,
            path,
            log_path=out_dir / f"{ds_name}.log.jsonl",
            max_examples=max_n,
            by_category=by_cat,
        )
        out_path = out_dir / f"{ds_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        _print_compact(ds_name, result)

    print(f"\nDone. Inspect:")
    print(f"  metrics:  {out_dir}/<dataset>.json")
    print(f"  per-instance log:  {out_dir}/<dataset>.log.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
