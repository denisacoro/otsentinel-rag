from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.generation.rag_pipeline import answer_question

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "query_logs"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the baseline RAG pipeline.")

    parser.add_argument("question")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--language", default="en", choices=["en", "ro"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config-name", default="section-bge-m3-512-64")

    return parser.parse_args()


def log_answer(answer) -> None:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIRECTORY / "queries.jsonl"

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(answer.model_dump_json() + "\n")


def main() -> None:
    arguments = parse_arguments()

    answer = answer_question(
        question=arguments.question,
        source_id=arguments.source_id,
        language=arguments.language,
        top_k=arguments.top_k,
        config_name=arguments.config_name,
    )

    log_answer(answer)

    print(f"\nQuestion: {answer.question}")
    print(f"Refused: {answer.refused}\n")
    print(f"Answer:\n{answer.answer_text}\n")
    print(f"Cited chunks: {answer.cited_chunk_ids}")
    print("\nRetrieved chunks:")
    for ref in answer.retrieved_chunks:
        print(f"  [{ref.score:.4f}] {ref.chunk_id} (pages {ref.page_start}-{ref.page_end})")
    print(
        f"\nRetrieval: {answer.retrieval_latency_ms:.0f}ms | "
        f"Generation: {answer.generation_latency_ms:.0f}ms | "
        f"Total: {answer.total_latency_ms:.0f}ms"
    )


if __name__ == "__main__":
    main()