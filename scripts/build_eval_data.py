#!/usr/bin/env python3
"""
Build eval_data.json from a Hugging Face text dataset.

Creates prompt/response pairs: prompt = sentence i, response = sentence i+1
within each document (story). Writes JSON list of {"prompt", "response"}.

Requires: pip install datasets

Example:
  python scripts/build_eval_data.py --max_pairs 20
  python scripts/build_eval_data.py --dataset roneneldan/TinyStories --max_pairs 50 --output eval_data.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterator


def split_sentences(text: str) -> list[str]:
    """Simple English sentence split (no NLTK)."""
    text = text.strip()
    if not text:
        return []
    # Split on . ! ? followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def iter_pairs_from_text(text: str, min_chars: int) -> Iterator[tuple[str, str]]:
    sents = split_sentences(text)
    for i in range(len(sents) - 1):
        a, b = sents[i], sents[i + 1]
        if len(a.strip()) >= min_chars and len(b.strip()) >= min_chars:
            yield (a.strip(), b.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval_data.json from HF dataset")
    parser.add_argument(
        "--dataset",
        default="roneneldan/TinyStories",
        help="Hugging Face dataset id (default: TinyStories)",
    )
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument(
        "--text_field",
        default="text",
        help="Column containing story/plain text (TinyStories uses 'text')",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=30,
        help="Maximum number of pairs to write",
    )
    parser.add_argument(
        "--max_stories",
        type=int,
        default=2000,
        help="Maximum stories to scan before stopping (speed limit)",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=15,
        help="Minimum characters per sentence for a pair to be kept",
    )
    parser.add_argument(
        "--output",
        default="eval_data.json",
        help="Output JSON path (relative to cwd)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: python -m pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=False)

    if args.text_field not in ds.column_names:
        print(f"Columns: {ds.column_names}", file=sys.stderr)
        raise SystemExit(f"Missing text field {args.text_field!r}")

    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []

    for row_idx in range(min(len(ds), args.max_stories)):
        text = ds[row_idx][args.text_field]
        if not isinstance(text, str):
            continue
        for prompt, response in iter_pairs_from_text(text, args.min_chars):
            key = (prompt, response)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"prompt": prompt, "response": response})
            if len(pairs) >= args.max_pairs:
                break
        if len(pairs) >= args.max_pairs:
            break

    if not pairs:
        raise SystemExit(
            "No pairs extracted. Check dataset field, split, or relax --min_chars."
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
