from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import mlflow

from app.core.settings import get_settings
from app.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse_embedder import SparseEmbedder
from app.retrieval.vector_store import (
    create_qdrant_client,
    search_dense_chunks,
    search_hybrid_chunks,
)
from app.schemas.eval import EvalQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "golden_eval_v1.jsonl"

METRIC_NAMES = ["recall_at_5", "recall_at_10", "precision_at_5", "mrr", "ndcg_at_10"]


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    questions = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                questions.append(EvalQuestion.model_validate_json(stripped_line))
    return questions


def compute_dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against the golden eval set.")

    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--config-name", default="section-bge-m3-512-64")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--run-name", default="dense-baseline")
    parser.add_argument("--search-mode", default="dense", choices=["dense", "hybrid"])
    parser.add_argument(
        "--prefetch-limit", type=int, default=25, help="Hybrid mode: candidates per branch before fusion."
    )
    parser.add_argument("--sparse-model-name", default="Qdrant/bm25")

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    settings = get_settings()

    all_questions = load_eval_questions(arguments.dataset_path)
    questions = [
        q for q in all_questions if q.split == arguments.split and q.answerable and q.relevant_chunk_ids
    ]

    if not questions:
        raise ValueError(f"No answerable questions found in split '{arguments.split}'.")

    print(
        f"Evaluating {len(questions)} answerable questions from the '{arguments.split}' split "
        f"using '{arguments.search_mode}' retrieval."
    )

    dense_embedder = DenseEmbedder(
        model_name=settings.embedding_model_name,
        requested_device=settings.embedding_device,
    )

    sparse_embedder = None
    if arguments.search_mode == "hybrid":
        sparse_embedder = SparseEmbedder(model_name=arguments.sparse_model_name)

    client = create_qdrant_client(settings.qdrant_url)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=arguments.run_name):
        mlflow.log_param("search_mode", arguments.search_mode)
        mlflow.log_param("config_name", arguments.config_name)
        mlflow.log_param("embedding_model", settings.embedding_model_name)

        if arguments.search_mode == "hybrid":
            mlflow.log_param("sparse_model", arguments.sparse_model_name)
            mlflow.log_param("prefetch_limit", arguments.prefetch_limit)
            mlflow.log_param("fusion", "RRF")

        mlflow.log_param("split", arguments.split)
        mlflow.log_param("max_k", arguments.max_k)
        mlflow.log_param("dataset_hash", compute_dataset_hash(arguments.dataset_path))
        mlflow.log_param("question_count", len(questions))

        per_question_rows = []

        for question in questions:
            relevant_ids = set(question.relevant_chunk_ids)
            source_id = question.relevant_document_ids[0] if question.relevant_document_ids else None

            start = time.perf_counter()

            if arguments.search_mode == "hybrid":
                dense_vector = dense_embedder.encode_query(question.question)
                sparse_vector = sparse_embedder.encode_query(question.question)

                results = search_hybrid_chunks(
                    client=client,
                    collection_name=settings.qdrant_collection,
                    dense_query_vector=dense_vector,
                    sparse_query_vector=sparse_vector,
                    top_k=arguments.max_k,
                    prefetch_limit=arguments.prefetch_limit,
                    config_name=arguments.config_name,
                    source_id=source_id,
                )
            else:
                query_vector = dense_embedder.encode_query(question.question)

                results = search_dense_chunks(
                    client=client,
                    collection_name=settings.qdrant_collection,
                    query_vector=query_vector,
                    top_k=arguments.max_k,
                    config_name=arguments.config_name,
                    source_id=source_id,
                )

            latency_ms = (time.perf_counter() - start) * 1000
            retrieved_ids = [str((r.payload or {}).get("chunk_id")) for r in results]

            row = {
                "question_id": question.question_id,
                "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
                "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
                "precision_at_5": precision_at_k(retrieved_ids, relevant_ids, 5),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
                "latency_ms": latency_ms,
            }
            per_question_rows.append(row)

            print(
                f"  {question.question_id}: recall@5={row['recall_at_5']:.2f} "
                f"recall@10={row['recall_at_10']:.2f} mrr={row['mrr']:.2f} "
                f"ndcg@10={row['ndcg_at_10']:.2f}"
            )

        for metric_name in METRIC_NAMES:
            values = [row[metric_name] for row in per_question_rows if row[metric_name] is not None]
            average = sum(values) / len(values) if values else 0.0
            mlflow.log_metric(metric_name, average)
            print(f"\nAverage {metric_name}: {average:.4f}")

        average_latency = sum(row["latency_ms"] for row in per_question_rows) / len(per_question_rows)
        mlflow.log_metric("avg_latency_ms", average_latency)
        print(f"Average retrieval latency: {average_latency:.1f}ms")


if __name__ == "__main__":
    main()