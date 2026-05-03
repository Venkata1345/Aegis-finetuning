"""Spot-check converted JSONL: random-sample N examples, verify span alignment.

For each sample, prints the source text and every span. The hard check is
that text[start:end] == span['text'] for every span — if any mismatch,
the converter is broken and we exit non-zero.

Usage:
    python -m data.spotcheck                  # 10 random examples from train
    python -m data.spotcheck --n 25
    python -m data.spotcheck --split validation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROC = Path(__file__).resolve().parent / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="train", choices=("train", "validation"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = PROC / f"{args.split}.jsonl"
    if not path.exists():
        print(f"Missing {path}. Run `python -m data.convert` first.", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    if not rows:
        print(f"{path} has 0 rows.", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    failures = 0
    for i, row in enumerate(sample, 1):
        text = row["text"]
        spans = row["spans"]
        print(f"\n=== Example {i}/{len(sample)} (id={row.get('id')}, lang={row.get('language')}) ===")
        print(f"TEXT ({len(text)} chars):")
        print(text)
        print(f"\nSPANS ({len(spans)}):")
        for span in spans:
            actual = text[span["start"] : span["end"]]
            ok = actual == span["text"]
            mark = "OK  " if ok else "FAIL"
            print(
                f"  [{mark}] {span['type']:11} "
                f"[{span['start']:>4}:{span['end']:>4}]  "
                f"claim={span['text']!r}  actual={actual!r}"
            )
            if not ok:
                failures += 1

    print(f"\n{'-' * 60}")
    print(f"Spot-check complete. {failures} mismatch(es) across {len(sample)} examples.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
