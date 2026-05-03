"""Eval harness — assemble the Markdown comparison table from predictor result JSONs.

The README in this repo is auto-regenerated from this output: no hand-edited
metrics, ever. Re-run after adding a new predictor or refreshing eval results
to update the comparison.

Reads:
    eval/results/<predictor>/main.json
    eval/results/<predictor>/heldout.json
    eval/results/<predictor>/adversarial.json
    eval/results/calibration/<predictor>/agreement.json   (optional)

Outputs:
    Markdown to stdout (default), to a file (--out), or injected into a
    target file between markers (--inject README.md).

Usage:
    python -m eval.harness                          # stdout, all predictors
    python -m eval.harness --predictors presidio    # restrict
    python -m eval.harness --inject README.md       # update the README
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DATASETS = ("main", "heldout", "adversarial")

INJECT_START = "<!-- AEGIS_TABLE_START -->"
INJECT_END = "<!-- AEGIS_TABLE_END -->"

PII_TYPES_ORDERED = (
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "DOB",
    "GOV_ID",
    "FINANCIAL",
    "MEDICAL_ID",
    "IP_ADDRESS",
)


# --- discovery / loading ---------------------------------------------------


def discover_predictors(results_dir: Path) -> list[str]:
    """Predictor subdirs of results_dir that have at least one *.json result."""
    if not results_dir.exists():
        return []
    found = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and d.name != "calibration" and any(d.glob("*.json")):
            found.append(d.name)
    return found


def load_predictor_results(results_dir: Path, predictor: str) -> dict:
    """Returns {dataset: result_dict, 'calibration': agreement_dict?}."""
    pdir = results_dir / predictor
    out: dict = {}
    for ds in DATASETS:
        path = pdir / f"{ds}.json"
        if path.exists():
            out[ds] = json.loads(path.read_text(encoding="utf-8"))
    cal_path = results_dir / "calibration" / predictor / "agreement.json"
    if cal_path.exists():
        out["calibration"] = json.loads(cal_path.read_text(encoding="utf-8"))
    return out


# --- section renderers -----------------------------------------------------


def _f1_or_dash(metric: dict | None) -> str:
    if not metric:
        return "—"
    if metric.get("tp", 0) + metric.get("fp", 0) + metric.get("fn", 0) == 0:
        return "—"
    return f"{metric.get('f1', 0):.3f}"


def _render_headline(data: dict[str, dict]) -> str:
    lines = [
        "## Headline (main test)",
        "",
        "| Predictor | Strict F1 | Schema valid | Hallucination | p50 latency | $/1k calls |",
        "|-----------|-----------|--------------|---------------|-------------|------------|",
    ]
    for pred, d in data.items():
        if "main" not in d:
            continue
        m = d["main"]
        f1 = m["strict_f1"]["overall"]["f1"]
        sv = m["schema_validity"]["rate"]
        h = m["hallucination"]["rate"]
        lat = m["perf"]["latency"]["p50_s"] * 1000
        cost = m["perf"]["cost"]
        cost_str = "self" if cost.get("self_hosted") else f"${cost.get('per_1k_usd', 0):.3f}"
        lines.append(
            f"| {pred} | {f1:.3f} | {sv * 100:.1f}% | {h * 100:.2f}% | {lat:.0f}ms | {cost_str} |"
        )
    return "\n".join(lines)


def _render_per_type(data: dict[str, dict]) -> str:
    predictors = [p for p, d in data.items() if "main" in d]
    if not predictors:
        return ""
    lines = ["## Strict F1 by PII type (main test)", ""]
    header = "| PII type | " + " | ".join(predictors) + " |"
    sep = "|" + "|".join(["---"] * (len(predictors) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for t in PII_TYPES_ORDERED:
        row = [t]
        for p in predictors:
            row.append(_f1_or_dash(data[p]["main"]["strict_f1"]["per_type"].get(t)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_adversarial_overall(data: dict[str, dict]) -> str:
    lines = [
        "## Adversarial (50 hand-written cases)",
        "",
        "| Predictor | Strict F1 | Partial F1 | Correctly empty | Hallucination |",
        "|-----------|-----------|------------|-----------------|---------------|",
    ]
    for pred, d in data.items():
        if "adversarial" not in d:
            continue
        a = d["adversarial"]
        sf = a["strict_f1"]["overall"]["f1"]
        pf = a["partial_f1"]["overall"]["f1"]
        ce = a["correctly_empty"]
        h = a["hallucination"]["rate"]
        ce_str = (
            f"{ce['rate'] * 100:.0f}% ({ce['correctly_empty_n']}/{ce['empty_gold_n']})"
            if ce.get("empty_gold_n", 0) > 0
            else "—"
        )
        lines.append(f"| {pred} | {sf:.3f} | {pf:.3f} | {ce_str} | {h * 100:.2f}% |")
    return "\n".join(lines)


def _render_adversarial_per_cat(data: dict[str, dict]) -> str:
    predictors = [
        p for p, d in data.items()
        if "adversarial" in d and "by_category" in d["adversarial"]
    ]
    if not predictors:
        return ""
    cats: set[str] = set()
    for p in predictors:
        cats.update(data[p]["adversarial"]["by_category"].keys())
    cats_sorted = sorted(cats)

    lines = ["## Per-adversarial-category strict F1", ""]
    lines.append(
        "_For categories where every case has empty gold "
        "(`reference_no_value`, `zero_pii_control`), the cell shows "
        "`correctly_empty` rate instead — strict F1 is meaningless on no-PII inputs._",
    )
    lines.append("")
    header = "| Category | " + " | ".join(predictors) + " |"
    sep = "|" + "|".join(["---"] * (len(predictors) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for cat in cats_sorted:
        row = [cat]
        for p in predictors:
            cat_data = data[p]["adversarial"]["by_category"].get(cat, {})
            n = cat_data.get("n", 0)
            ce = cat_data.get("correctly_empty", {})
            sf = cat_data.get("strict_f1", {})
            if n == 0:
                row.append("—")
            elif ce.get("empty_gold_n", 0) == n:
                row.append(f"empty {ce['rate'] * 100:.0f}%")
            else:
                row.append(f"{sf.get('f1', 0):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_held_out(data: dict[str, dict]) -> str:
    lines = [
        "## Held-out (50 Wikipedia carriers + injected PII)",
        "",
        "_Ground truth = injected PII only; precision is structurally biased "
        "downward because predictors correctly find pre-existing Wikipedia PII that "
        "isn't in the injection-only ground truth. **Recall on injected PII** "
        "is the meaningful metric on this set._",
        "",
        "| Predictor | Recall on injected | Strict F1 | n FP | n injected |",
        "|-----------|---------------------|-----------|------|------------|",
    ]
    for pred, d in data.items():
        if "heldout" not in d:
            continue
        sf = d["heldout"]["strict_f1"]["overall"]
        recall = sf["recall"]
        f1 = sf["f1"]
        fp = sf["fp"]
        injected = sf["tp"] + sf["fn"]
        lines.append(
            f"| {pred} | {recall * 100:.1f}% | {f1:.3f} | {fp} | {injected} |"
        )
    return "\n".join(lines)


def _render_calibration(data: dict[str, dict]) -> str:
    cal_preds = [p for p, d in data.items() if "calibration" in d]
    if not cal_preds:
        return ""
    lines = [
        "## Calibration (cross-LLM-judge agreement)",
        "",
        "_No human gold standard — two independent LLM judges score the same "
        "sample on a 1-5 rubric; reported r is the Pearson correlation between "
        "their scores. Target r ≥ 0.6; below that, the rubric is iterated._",
        "",
        "| Predictor | Judge A | Judge B | n | Pearson r | Passes 0.6 target? |",
        "|-----------|---------|---------|---|-----------|--------------------|",
    ]
    for p in cal_preds:
        c = data[p]["calibration"]
        corr = c.get("correlation", {})
        r = corr.get("r")
        if r is None:
            r_str = f"— _{corr.get('note', 'n/a')}_"
        else:
            r_str = f"{r:.3f}"
        passes = "yes" if c.get("passes_target") else "no"
        lines.append(
            f"| {p} | {c.get('judge_a', '—')} | {c.get('judge_b', '—')} "
            f"| {c.get('n_aligned', 0)} | {r_str} | {passes} |"
        )
    return "\n".join(lines)


# --- top-level assembly ---------------------------------------------------


def render_markdown(data: dict[str, dict]) -> str:
    if not data:
        return "_No predictor results found in eval/results/._"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts: list[str] = [
        f"<!-- generated by eval.harness at {timestamp} -->",
        f"<!-- predictors: {', '.join(sorted(data.keys()))} -->",
        "",
    ]
    for renderer in (
        _render_headline,
        _render_per_type,
        _render_adversarial_overall,
        _render_adversarial_per_cat,
        _render_held_out,
        _render_calibration,
    ):
        section = renderer(data)
        if section:
            parts.append(section)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# --- README injection -----------------------------------------------------


def inject_into_file(target: Path, markdown: str) -> bool:
    """Replace content between AEGIS_TABLE markers in `target`. Idempotent."""
    if not target.exists():
        return False
    content = target.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(INJECT_START) + r".*?" + re.escape(INJECT_END),
        re.DOTALL,
    )
    if not pattern.search(content):
        return False
    replacement = f"{INJECT_START}\n{markdown}{INJECT_END}"
    target.write_text(pattern.sub(replacement, content), encoding="utf-8")
    return True


# --- CLI ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument(
        "--predictors",
        nargs="*",
        default=None,
        help="Restrict to these predictor names; default: auto-discover.",
    )
    ap.add_argument("--out", default=None, help="Write Markdown to this file.")
    ap.add_argument(
        "--inject",
        default=None,
        help=(
            "Replace content between AEGIS_TABLE markers in this file. "
            "Use to keep README.md auto-updated."
        ),
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    predictors = args.predictors or discover_predictors(results_dir)
    if not predictors:
        print(f"No predictor results found under {results_dir}.", file=sys.stderr)
        return 1
    data = {p: load_predictor_results(results_dir, p) for p in predictors}
    md = render_markdown(data)

    if args.inject:
        target = Path(args.inject)
        if inject_into_file(target, md):
            print(f"Injected into {target}", file=sys.stderr)
        else:
            print(
                f"Could not inject (file missing or markers absent): {target}",
                file=sys.stderr,
            )
            return 2
    elif args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
