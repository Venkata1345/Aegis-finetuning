"""Build the held-out eval set: Wikipedia bio carriers + synthetic PII injection.

For each of N=50 Wikipedia article first paragraphs, inject 1-4 synthetic
PII items via Faker at random word boundaries and record exact char offsets
as ground truth.

CAVEAT (documented in README): ground truth includes ONLY injected PII.
Pre-existing PII in the Wikipedia carrier text (dates, places of birth for
public figures) may inflate false-positive counts on this set. The
adversarial set is the cleaner FP measure; the main test split
(data/processed/test.chat.jsonl) provides full-distribution ground truth
from ai4privacy.

Network: Wikipedia summaries are cached at data/eval/wikipedia_cache.json
so re-runs are deterministic and offline. Delete the cache to refresh.

Output: data/eval/heldout.jsonl  (one JSON object per line, schema below)

  {"id": "heldout_marie_curie",
   "carrier": "Marie Curie",
   "text": "...full text with injected PII...",
   "spans": [{"type": "EMAIL", "start": 312, "end": 333, "text": "..."}, ...]}

Run: python -m data.heldout
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import wikipedia
from faker import Faker

OUT_DIR = Path(__file__).resolve().parent / "eval"
OUT_PATH = OUT_DIR / "heldout.jsonl"
CACHE_PATH = OUT_DIR / "wikipedia_cache.json"

SEED = 42
TARGET_N = 50
SUMMARY_SENTENCES = 5
MIN_CARRIER_CHARS = 250
FETCH_SLEEP_S = 0.5  # gentle rate-limit between fresh Wikipedia fetches
MAX_RETRIES = 1      # retry once on JSONDecodeError (Wikipedia rate-limit signature)

# Curated public-figure list. Diverse fields/eras; first-paragraph summaries
# tend to be descriptive bio prose. Order is the seed for our 50; we stop
# once we have TARGET_N successful summaries.
WIKI_TITLES: list[str] = [
    # Sciences / math
    "Marie Curie", "Charles Darwin", "Alan Turing", "Ada Lovelace",
    "Isaac Newton", "Albert Einstein", "Galileo Galilei", "Hypatia",
    "Nikola Tesla", "Michael Faraday", "Dmitri Mendeleev", "Stephen Hawking",
    "Linus Pauling", "Richard Feynman", "Barbara McClintock", "Rosalind Franklin",
    "Carl Friedrich Gauss", "Leonhard Euler", "Srinivasa Ramanujan", "Paul Erdős",
    "David Hilbert", "Emmy Noether", "Subrahmanyan Chandrasekhar", "Lise Meitner",
    "Grace Hopper", "Edsger W. Dijkstra", "Donald Knuth", "John von Neumann",
    "Claude Shannon",
    # Writers / arts
    "Jane Austen", "Charles Dickens", "Leo Tolstoy", "William Shakespeare",
    "Charlotte Brontë", "Chinua Achebe", "Rabindranath Tagore", "Jorge Luis Borges",
    "Toni Morrison", "Gabriel García Márquez", "Virginia Woolf",
    "Leonardo da Vinci", "Michelangelo", "Frida Kahlo", "Georgia O'Keeffe",
    "Pablo Picasso", "Katsushika Hokusai",
    "Johann Sebastian Bach", "Ludwig van Beethoven", "Wolfgang Amadeus Mozart",
    "Felix Mendelssohn", "Gustav Mahler",
    # Philosophy
    "Immanuel Kant", "Bertrand Russell", "Ludwig Wittgenstein", "Aristotle",
    "Confucius", "Hannah Arendt", "Simone de Beauvoir", "Friedrich Nietzsche",
    "John Stuart Mill",
    # History / activism
    "Abraham Lincoln", "Nelson Mandela", "Mahatma Gandhi", "Cleopatra",
    "Akbar", "Joan of Arc", "Sojourner Truth", "Frederick Douglass",
    "Harriet Tubman", "Susan B. Anthony", "Emmeline Pankhurst",
    "W. E. B. Du Bois", "Olaudah Equiano", "Rachel Carson", "Edward Said",
    "Amartya Sen", "Wangari Maathai",
    # Top-up batch (added 2026-05-03 after first run hit Wikipedia rate-limit
    # past the 47th title; gives margin for further failures).
    "Erwin Schrödinger", "Niels Bohr", "Werner Heisenberg", "Carl Sagan",
    "Tim Berners-Lee", "Linus Torvalds", "Hedy Lamarr", "Maryam Mirzakhani",
    "Katherine Johnson", "Charles Babbage", "Edmond Halley",
    "Pierre-Simon Laplace", "Henrietta Leavitt", "Mary Anning", "Vera Rubin",
]


def _load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_summary(title: str, cache: dict[str, str], attempt: int = 0) -> str | None:
    if title in cache:
        return cache[title]
    # Sleep BEFORE the request — gentle on Wikipedia, with backoff on retry.
    time.sleep(FETCH_SLEEP_S * (1 + 4 * attempt))
    try:
        text = wikipedia.summary(title, sentences=SUMMARY_SENTENCES, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError:
        print(f"  skip '{title}': disambiguation", file=sys.stderr)
        return None
    except wikipedia.exceptions.PageError:
        print(f"  skip '{title}': page not found", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        if attempt < MAX_RETRIES:
            print(f"  retry '{title}' (attempt {attempt + 2}): {e}", file=sys.stderr)
            return fetch_summary(title, cache, attempt + 1)
        print(f"  skip '{title}': JSONDecodeError after {MAX_RETRIES + 1} attempts", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  skip '{title}': {type(e).__name__}: {e}", file=sys.stderr)
        return None
    cache[title] = text
    return text


def build_generators(fake: Faker) -> list[tuple[str, Callable[[], str]]]:
    """(category, generator) pairs spanning all 9 PII types.
    Multiple entries per type let us bias the random sampling toward variety.
    """
    return [
        ("PERSON", fake.name),
        ("PERSON", fake.name),
        ("EMAIL", fake.email),
        ("EMAIL", fake.email),
        ("PHONE", fake.phone_number),
        ("PHONE", fake.phone_number),
        ("ADDRESS", lambda: fake.address().replace("\n", ", ")),
        ("ADDRESS", lambda: fake.address().replace("\n", ", ")),
        ("DOB", lambda: fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%d %B %Y")),
        ("DOB", lambda: fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%Y-%m-%d")),
        ("GOV_ID", fake.ssn),
        ("FINANCIAL", fake.credit_card_number),
        ("FINANCIAL", fake.iban),
        ("IP_ADDRESS", fake.ipv4),
        ("IP_ADDRESS", fake.ipv6),
        ("MEDICAL_ID", lambda: f"MRN-{fake.random_number(digits=7, fix_len=True)}"),
    ]


def inject_pii(
    text: str,
    gens: list[tuple[str, Callable[[], str]]],
    rng: random.Random,
    k_min: int = 1,
    k_max: int = 4,
) -> tuple[str, list[dict]]:
    """Insert k random PII items at random whitespace boundaries.

    Returns (new_text, spans). Asserts every span's text matches new_text[start:end].
    """
    boundaries = [i for i, ch in enumerate(text) if ch == " "]
    if not boundaries:
        return text, []
    k = rng.randint(k_min, min(k_max, len(boundaries)))
    insert_positions = sorted(rng.sample(boundaries, k))

    out_chunks: list[str] = []
    spans: list[dict] = []
    last = 0
    cum_offset = 0
    for orig_pos in insert_positions:
        type_, gen = rng.choice(gens)
        value = gen()
        out_chunks.append(text[last:orig_pos])
        out_chunks.append(" ")
        span_start = orig_pos + cum_offset + 1
        out_chunks.append(value)
        span_end = span_start + len(value)
        spans.append({"type": type_, "start": span_start, "end": span_end, "text": value})
        last = orig_pos
        cum_offset += 1 + len(value)
    out_chunks.append(text[last:])
    new_text = "".join(out_chunks)

    for s in spans:
        actual = new_text[s["start"] : s["end"]]
        if actual != s["text"]:
            raise AssertionError(f"Span mismatch: claim {s['text']!r}, actual {actual!r}")
    return new_text, spans


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()

    fake = Faker("en_US")
    Faker.seed(SEED)
    rng = random.Random(SEED)
    gens = build_generators(fake)

    rows: list[dict] = []
    for title in WIKI_TITLES:
        if len(rows) >= TARGET_N:
            break
        summary = fetch_summary(title, cache)
        if not summary or len(summary) < MIN_CARRIER_CHARS:
            continue
        carrier = re.sub(r"\s+", " ", summary).strip()
        new_text, spans = inject_pii(carrier, gens, rng)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        rows.append({
            "id": f"heldout_{slug}",
            "carrier": title,
            "text": new_text,
            "spans": spans,
        })

    _save_cache(cache)

    if len(rows) < TARGET_N:
        print(f"Warning: only got {len(rows)} bios (target {TARGET_N})", file=sys.stderr)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} held-out examples to {OUT_PATH}")
    type_counts: dict[str, int] = {}
    for row in rows:
        for s in row["spans"]:
            type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
    total_spans = sum(type_counts.values())
    print(f"Total injected PII: {total_spans} across {len(rows)} carriers")
    for t in sorted(type_counts):
        print(f"  {t:11} {type_counts[t]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
