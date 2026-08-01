from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from app.schemas.eval import EvalQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "golden_eval_v1.jsonl"


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    questions = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                questions.append(EvalQuestion.model_validate_json(stripped_line))
            except Exception as error:
                raise ValueError(f"Invalid question on line {line_number}: {error}") from error

    return questions


def compute_dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the golden evaluation dataset.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    arguments = parser.parse_args()

    questions = load_eval_questions(arguments.dataset_path)

    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question_id values found.")

    for question in questions:
        if question.answerable and not question.relevant_chunk_ids:
            raise ValueError(f"{question.question_id} is answerable but has no relevant_chunk_ids.")
        if not question.answerable and question.relevant_chunk_ids:
            raise ValueError(f"{question.question_id} is unanswerable but lists relevant_chunk_ids.")

    total = len(questions)
    unanswerable = sum(1 for question in questions if not question.answerable)
    languages = Counter(question.language for question in questions)
    types = Counter(question.question_type for question in questions)
    splits = Counter(question.split for question in questions)
    documents = Counter(
        document_id for question in questions for document_id in question.relevant_document_ids
    )

    print(f"Total questions: {total}")
    print(f"Unanswerable: {unanswerable} ({unanswerable / total:.1%})")
    print(f"Languages: {dict(languages)}")
    print(f"Question types: {dict(types)}")
    print(f"Splits: {dict(splits)}")
    print(f"Documents referenced: {dict(documents)}")
    print(f"\nDataset SHA-256: {compute_dataset_hash(arguments.dataset_path)}")


if __name__ == "__main__":
    main()