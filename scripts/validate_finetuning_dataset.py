"""Quality-check the fine-tuning dataset before it is used for QLoRA training.

Checks performed:
  1. Schema validation against app.schemas.finetuning.FinetuningExample.
  2. Duplicate example_id detection.
  3. Near-duplicate assistant-answer detection (text-similarity based -- see note below).
  4. Leakage check: no source_chunk_id may overlap with golden_eval_v1.jsonl's
     relevant_chunk_ids (those are reserved for evaluation, never training).
  5. Citation-tag consistency: every [Sx] tag cited in the assistant answer must be
     one of the tags defined in the Context section of the user message (no invented
     tags), and every non-refusal example must cite at least one tag.
  6. Refusal-text exact-match: every refusal_insufficient_context example's assistant
     message must equal the live system's REFUSAL_TEXT constant exactly (verbatim,
     regardless of question language -- the live system does not localize it).

Note on near-duplicate detection: this uses stdlib difflib.SequenceMatcher over the
assistant answer text, not embedding-based semantic similarity. It will catch near-
identical phrasing but not paraphrases with different wording. Swap in DenseEmbedder
+ cosine similarity later if that's not sensitive enough in practice.

Usage:
    python scripts/validate_finetuning_dataset.py
    python scripts/validate_finetuning_dataset.py --dataset-path data/finetuning/train_v1.jsonl
    python scripts/validate_finetuning_dataset.py --similarity-threshold 0.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from app.schemas.eval import EvalQuestion
from app.schemas.finetuning import FinetuningExample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "finetuning" / "train_v1.jsonl"
DEFAULT_GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "golden_eval_v1.jsonl"

CITATION_TAG_PATTERN = re.compile(r"\[S(\d+)\]")
CONTEXT_TAG_PATTERN = re.compile(r"^\[S(\d+)\]", flags=re.MULTILINE)
REFUSAL_TEXT = "I don't have enough evidence in the current knowledge base to answer that."
DEFAULT_SIMILARITY_THRESHOLD = 0.90


def load_finetuning_examples(path: Path) -> list[FinetuningExample]:
    examples: list[FinetuningExample] = []
    with path.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON -- {exc}") from exc
            try:
                examples.append(FinetuningExample.model_validate(raw))
            except Exception as exc:
                raise ValueError(
                    f"{path.name}:{line_number}: schema validation failed -- {exc}"
                ) from exc
    return examples


def load_reserved_chunk_ids(golden_eval_path: Path) -> set[str]:
    reserved: set[str] = set()
    with golden_eval_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            question = EvalQuestion.model_validate(json.loads(line))
            reserved.update(question.relevant_chunk_ids)
    return reserved


def compute_dataset_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def check_duplicate_example_ids(examples: list[FinetuningExample]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for ex in examples:
        if ex.example_id in seen:
            duplicates.append(ex.example_id)
        seen.add(ex.example_id)
    return duplicates


def check_near_duplicate_answers(
    examples: list[FinetuningExample], threshold: float
) -> list[tuple[str, str, float]]:
    """Flag pairs of examples whose assistant answers are near-identical text.

    O(n^2) over answer text -- fine at hundreds of examples, would need a smarter
    approach (embeddings + ANN) at thousands.
    """
    flagged: list[tuple[str, str, float]] = []
    answers = [
        (ex.example_id, ex.messages[-1].content.strip())
        for ex in examples
        if ex.task_type != "refusal_insufficient_context"
    ]
    for i in range(len(answers)):
        id_a, text_a = answers[i]
        for j in range(i + 1, len(answers)):
            id_b, text_b = answers[j]
            ratio = SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= threshold:
                flagged.append((id_a, id_b, round(ratio, 3)))
    return flagged


def check_leakage(
    examples: list[FinetuningExample], reserved_chunk_ids: set[str]
) -> list[tuple[str, list[str]]]:
    leaks: list[tuple[str, list[str]]] = []
    for ex in examples:
        overlap = set(ex.source_chunk_ids) & reserved_chunk_ids
        if overlap:
            leaks.append((ex.example_id, sorted(overlap)))
    return leaks


def extract_context_tags(user_message: str) -> set[str]:
    return set(CONTEXT_TAG_PATTERN.findall(user_message))


def check_citation_consistency(examples: list[FinetuningExample]) -> list[tuple[str, str, list[str]]]:
    """Returns a list of (example_id, issue_type, detail) tuples.

    issue_type is one of: "invented_tag", "no_citation".
    """
    issues: list[tuple[str, str, list[str]]] = []
    for ex in examples:
        if ex.task_type == "refusal_insufficient_context":
            continue
        user_msg = next(m.content for m in ex.messages if m.role == "user")
        assistant_msg = next(m.content for m in ex.messages if m.role == "assistant")
        context_tags = extract_context_tags(user_msg)
        cited_tags = set(CITATION_TAG_PATTERN.findall(assistant_msg))
        invented = cited_tags - context_tags
        if invented:
            issues.append((ex.example_id, "invented_tag", sorted(invented)))
        if not cited_tags:
            issues.append((ex.example_id, "no_citation", []))
    return issues


def check_refusal_text(examples: list[FinetuningExample]) -> list[str]:
    issues: list[str] = []
    for ex in examples:
        if ex.task_type != "refusal_insufficient_context":
            continue
        assistant_msg = next(m.content for m in ex.messages if m.role == "assistant")
        if assistant_msg.strip() != REFUSAL_TEXT:
            issues.append(ex.example_id)
    return issues


def check_message_roles(examples: list[FinetuningExample]) -> list[str]:
    """Every example should be system, user, assistant in that order (min_length=3
    is already enforced by the schema; this checks the actual role sequence)."""
    issues: list[str] = []
    for ex in examples:
        roles = [m.role for m in ex.messages]
        if roles[:3] != ["system", "user", "assistant"]:
            issues.append(ex.example_id)
    return issues


def print_stats(examples: list[FinetuningExample]) -> None:
    print("\n--- Dataset composition ---")
    print(f"Total examples: {len(examples)}")
    print(f"By task_type: {dict(Counter(ex.task_type for ex in examples))}")
    print(f"By split: {dict(Counter(ex.split for ex in examples))}")
    print(f"By language: {dict(Counter(ex.language for ex in examples))}")
    source_counts: Counter[str] = Counter()
    for ex in examples:
        source_counts.update(ex.source_document_ids)
    print(f"By source document: {dict(source_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--golden-eval-path", type=Path, default=DEFAULT_GOLDEN_EVAL_PATH)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="difflib ratio above which two answers are flagged as near-duplicates (0-1)",
    )
    args = parser.parse_args()

    if not args.dataset_path.exists():
        print(f"ERROR: dataset not found at {args.dataset_path}", file=sys.stderr)
        raise SystemExit(2)
    if not args.golden_eval_path.exists():
        print(f"ERROR: golden eval set not found at {args.golden_eval_path}", file=sys.stderr)
        raise SystemExit(2)

    examples = load_finetuning_examples(args.dataset_path)
    reserved_chunk_ids = load_reserved_chunk_ids(args.golden_eval_path)

    print(f"Loaded {len(examples)} fine-tuning examples from {args.dataset_path}")
    print(f"Dataset SHA-256: {compute_dataset_hash(args.dataset_path)}")
    print(f"Reserved (golden-eval) chunk IDs excluded from training: {len(reserved_chunk_ids)}")

    print_stats(examples)

    print("\n--- Quality checks ---")
    failed = False

    dup_ids = check_duplicate_example_ids(examples)
    if dup_ids:
        failed = True
        print(f"FAIL duplicate example_id: {dup_ids}")
    else:
        print("PASS no duplicate example_id")

    role_issues = check_message_roles(examples)
    if role_issues:
        failed = True
        print(f"FAIL unexpected message role order: {role_issues}")
    else:
        print("PASS all examples are system/user/assistant")

    leaks = check_leakage(examples, reserved_chunk_ids)
    if leaks:
        failed = True
        print(f"FAIL golden-eval leakage in {len(leaks)} example(s):")
        for example_id, chunk_ids in leaks:
            print(f"  {example_id}: {chunk_ids}")
    else:
        print("PASS no golden-eval chunk leakage")

    citation_issues = check_citation_consistency(examples)
    if citation_issues:
        failed = True
        print(f"FAIL citation issues in {len(citation_issues)} example(s):")
        for example_id, issue_type, detail in citation_issues:
            print(f"  {example_id}: {issue_type} {detail}")
    else:
        print("PASS all non-refusal examples cite only tags present in their context")

    refusal_issues = check_refusal_text(examples)
    if refusal_issues:
        failed = True
        print(f"FAIL refusal text mismatch in: {refusal_issues}")
    else:
        print("PASS all refusal examples use the exact REFUSAL_TEXT string")

    near_dupes = check_near_duplicate_answers(examples, args.similarity_threshold)
    if near_dupes:
        # Warning, not a hard failure -- some legitimate reuse of source chunks
        # across different questions can produce similar-but-not-identical answers.
        print(
            f"WARN {len(near_dupes)} near-duplicate answer pair(s) "
            f"(threshold={args.similarity_threshold}), review manually:"
        )
        for id_a, id_b, ratio in near_dupes:
            print(f"  {id_a} <-> {id_b}: similarity={ratio}")
    else:
        print(f"PASS no near-duplicate answers above threshold={args.similarity_threshold}")

    print()
    if failed:
        print("RESULT: FAILED -- fix the issues above before using this dataset for training.")
        raise SystemExit(1)
    print("RESULT: PASSED -- dataset is ready for training (review any WARNs above manually).")


if __name__ == "__main__":
    main()
