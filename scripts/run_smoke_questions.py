from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.generation.rag_pipeline import answer_question

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "smoke_questions.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "smoke_results" / "latest.jsonl"


def load_smoke_questions(path: Path) -> list[dict]:
    questions = []

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                questions.append(json.loads(stripped_line))

    return questions


def evaluate_result(question: dict, answer) -> str:
    """Score one answer against its expected answerability."""

    if question["answerable"] and answer.refused:
        return "FAIL (refused an answerable question)"

    if not question["answerable"] and not answer.refused:
        return "FAIL (answered an unanswerable question)"

    if question["answerable"] and not answer.refused and not answer.cited_chunk_ids:
        return "REVIEW (answered but no valid citation found)"

    return "PASS"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manual smoke-question suite.")

    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    questions = load_smoke_questions(arguments.questions_path)
    arguments.results_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with arguments.results_path.open("w", encoding="utf-8") as results_file:
        for question in questions:
            print(f"\n[{question['question_id']}] {question['question']}")

            answer = answer_question(
                question=question["question"],
                source_id=question.get("source_id"),
                language=question.get("language", "en"),
            )

            verdict = evaluate_result(question, answer)
            print(f"  -> {verdict}")
            print(f"  Refused: {answer.refused} | Cited: {answer.cited_chunk_ids}")

            results.append(
                {
                    "question_id": question["question_id"],
                    "question_type": question["question_type"],
                    "verdict": verdict,
                }
            )

            results_file.write(
                json.dumps(
                    {
                        "question_id": question["question_id"],
                        "verdict": verdict,
                        "answer": answer.model_dump(mode="json"),
                    }
                )
                + "\n"
            )

    passed_count = sum(1 for result in results if result["verdict"] == "PASS")

    print(f"\n{passed_count}/{len(results)} passed. Full results: {arguments.results_path}")


if __name__ == "__main__":
    main()