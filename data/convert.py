"""Convert raw ai4privacy JSONL into Aegis schema JSONL.

ai4privacy provides character-offset spans directly via the `privacy_mask`
field, so each example becomes:

    {"id": "...", "language": "...", "text": "...",
     "spans": [{"type": "PERSON", "start": 14, "end": 24, "text": "John Smith"}, ...]}

Defensive checks (each is counted in the dropped report):
  - empty text
  - bad offsets (non-int or end<=start)
  - out-of-bounds offsets
  - value mismatch (text[start:end] != claimed value)
  - schema_invalid (Pydantic rejection)

Source labels not in data.labels._AI4P_TO_AEGIS are SKIPPED and counted
under "unmapped". The unmapped report is the signal for updating
data/labels.py — run with --strict to abort on the first unmapped label
once you believe the mapping is complete.

Usage:
    python -m data.convert                       # English-only, full
    python -m data.convert --max-rows 1000       # quick smoke test
    python -m data.convert --lang all            # keep all languages
    python -m data.convert --strict              # fail on unmapped labels
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from data.labels import map_label
from data.schema import Span

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "processed"


def _coerce_privacy_mask(raw: object) -> list[dict]:
    """privacy_mask is sometimes list-of-dicts, sometimes a JSON string. Normalise."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict)]


def convert_row(row: dict, unmapped: Counter, dropped: Counter) -> dict | None:
    text = row.get("source_text")
    if not isinstance(text, str) or not text:
        dropped["empty_text"] += 1
        return None

    spans: list[dict] = []
    for entry in _coerce_privacy_mask(row.get("privacy_mask")):
        label = entry.get("label")
        start = entry.get("start")
        end = entry.get("end")
        value = entry.get("value")

        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            dropped["bad_offsets"] += 1
            continue
        if start < 0 or end > len(text):
            dropped["out_of_bounds"] += 1
            continue
        if isinstance(value, str) and text[start:end] != value:
            dropped["value_mismatch"] += 1
            continue

        aegis_type = map_label(label if isinstance(label, str) else None)
        if aegis_type is None:
            unmapped[label if isinstance(label, str) else "<empty>"] += 1
            continue

        try:
            span = Span(type=aegis_type, start=start, end=end, text=text[start:end])
        except ValidationError:
            dropped["schema_invalid"] += 1
            continue
        spans.append(span.model_dump())

    spans.sort(key=lambda s: (s["start"], s["end"]))
    return {
        "id": row.get("id"),
        "language": row.get("language"),
        "text": text,
        "spans": spans,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lang",
        default="English",
        help="Language filter ('English', 'all', or a specific language string). Default: English.",
    )
    ap.add_argument("--strict", action="store_true", help="Fail if any unmapped labels are seen.")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit input rows per split.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print(f"No raw JSONL in {RAW_DIR}. Run `python -m data.download` first.", file=sys.stderr)
        return 1

    unmapped: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    total_in = 0
    total_out = 0

    for raw in raw_files:
        split = raw.stem
        out_path = OUT_DIR / f"{split}.jsonl"
        in_count = 0
        out_count = 0
        with raw.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                in_count += 1
                if args.max_rows and in_count > args.max_rows:
                    break
                row = json.loads(line)
                lang = row.get("language", "")
                if args.lang != "all" and lang != args.lang:
                    continue
                converted = convert_row(row, unmapped, dropped)
                if converted is None:
                    continue
                f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                out_count += 1
        print(f"  {split:>10}: {in_count:>7} read, {out_count:>7} written -> {out_path}")
        total_in += in_count
        total_out += out_count

    print(f"\nTotal: {total_in} read, {total_out} written")
    if dropped:
        print("\nDropped spans (per reason):")
        for reason, count in dropped.most_common():
            print(f"  {reason}: {count}")
    if unmapped:
        print("\nUNMAPPED source labels (top 30) — review and add to data/labels.py:")
        for lbl, count in unmapped.most_common(30):
            print(f"  {lbl}: {count}")
        if args.strict:
            print("\n--strict: failing because unmapped labels were seen.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
