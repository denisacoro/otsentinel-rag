"""Audit token lengths across the full fine-tuning dataset.

The feasibility test only sampled the first 8 examples (max 711 tokens). Before
picking max_length for the real training run, we need the true distribution across
all 54 examples -- the two-chunk comparison examples in particular are likely
longer. Silently truncating an example during training would cut off part of the
context or the answer, which teaches the model something subtly wrong rather than
just failing loudly, so it's worth checking before committing to a max_length.

This only loads the tokenizer (already cached locally from the feasibility test),
not the full model, so it runs in a few seconds with no GPU involved.

Usage:
    python scripts/audit_finetuning_lengths.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "finetuning" / "train_v1.jsonl"
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows: list[dict] = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    lengths: list[tuple[str, str, int]] = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        token_count = len(tokenizer(text).input_ids)
        lengths.append((row["example_id"], row["task_type"], token_count))

    lengths.sort(key=lambda x: x[2], reverse=True)
    token_counts = [t for _, _, t in lengths]

    print(f"Examples: {len(lengths)}")
    print(f"Min: {min(token_counts)}  Max: {max(token_counts)}  "
          f"Mean: {statistics.mean(token_counts):.0f}  "
          f"Median: {statistics.median(token_counts):.0f}")
    print(f"95th percentile: {sorted(token_counts)[int(len(token_counts) * 0.95)]}")

    print("\nTop 10 longest examples:")
    for example_id, task_type, token_count in lengths[:10]:
        print(f"  {example_id} ({task_type}): {token_count} tokens")

    print("\nToken budget check at common max_length values:")
    for candidate in (768, 1024, 1280, 1536, 2048):
        over = sum(1 for t in token_counts if t > candidate)
        print(f"  max_length={candidate}: {over}/{len(token_counts)} example(s) would be truncated")


if __name__ == "__main__":
    main()