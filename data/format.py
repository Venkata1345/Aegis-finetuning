"""Wrap converted spans as instruction-tuned chat messages for training.

Reads data/processed/{train,validation}.jsonl (output of data/convert) and
emits chat-format JSONL ready for Unsloth's SFT trainer.

Aegis splits (deterministic, seed=42):
  source train (lang-filtered) -> 95% Aegis train, 5% Aegis val (early stopping)
  source validation            -> Aegis test (held-out; never seen during training)

Usage:
    python -m data.format
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from inference.prompts import SYSTEM_PROMPT

PROC = Path(__file__).resolve().parent / "processed"
SEED = 42
VAL_FRACTION = 0.05


def to_chat(text: str, spans: list[dict]) -> dict:
    """Build a {"messages": [...]} record for SFT."""
    cleaned = [
        {"type": s["type"], "start": s["start"], "end": s["end"], "text": s["text"]}
        for s in spans
    ]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(cleaned, ensure_ascii=False)},
        ]
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    argparse.ArgumentParser().parse_args()

    src_train = PROC / "train.jsonl"
    src_val = PROC / "validation.jsonl"
    if not src_train.exists() or not src_val.exists():
        print("Missing converted JSONL. Run `python -m data.convert` first.", file=sys.stderr)
        return 1

    train_rows = _read_jsonl(src_train)
    test_rows = _read_jsonl(src_val)

    rng = random.Random(SEED)
    rng.shuffle(train_rows)
    cut = int(len(train_rows) * (1 - VAL_FRACTION))
    aegis_train, aegis_val = train_rows[:cut], train_rows[cut:]
    aegis_test = test_rows

    _write_jsonl([to_chat(r["text"], r["spans"]) for r in aegis_train], PROC / "train.chat.jsonl")
    _write_jsonl([to_chat(r["text"], r["spans"]) for r in aegis_val], PROC / "val.chat.jsonl")
    _write_jsonl([to_chat(r["text"], r["spans"]) for r in aegis_test], PROC / "test.chat.jsonl")

    print(f"  train: {len(aegis_train):>7} rows -> train.chat.jsonl")
    print(f"  val:   {len(aegis_val):>7} rows -> val.chat.jsonl")
    print(f"  test:  {len(aegis_test):>7} rows -> test.chat.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
