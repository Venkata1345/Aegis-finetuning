"""Download ai4privacy/pii-masking-300k from Hugging Face Datasets.

Writes one JSONL per split to data/raw/. The BIO -> char-offset conversion
is NOT needed: ai4privacy provides char-offset spans directly via the
`privacy_mask` field. Run `python -m data.convert` next.

Smoke-test mode: --max-rows N uses streaming so only N rows per split are
fetched (no full ~500MB cache). Useful for verifying the pipeline before
committing to the full download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "ai4privacy/pii-masking-300k"
RAW_DIR = Path(__file__).resolve().parent / "raw"
SPLITS = ("train", "validation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stream and stop after N rows per split (smoke test, no full cache).",
    )
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_NAME} from Hugging Face Hub...")

    if args.max_rows:
        for split in SPLITS:
            stream = load_dataset(DATASET_NAME, split=split, streaming=True)
            out_path = RAW_DIR / f"{split}.jsonl"
            written = 0
            with out_path.open("w", encoding="utf-8") as f:
                for row in stream:
                    if written >= args.max_rows:
                        break
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
            print(f"  {split}: {written} examples (streamed) -> {out_path}")
        return 0

    ds = load_dataset(DATASET_NAME)
    for split, split_ds in ds.items():
        out_path = RAW_DIR / f"{split}.jsonl"
        split_ds.to_json(str(out_path), lines=True)
        print(f"  {split}: {len(split_ds):>7} examples -> {out_path}")

    print("\nDone. Next: `python -m data.convert` to build char-offset span JSONL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
