from __future__ import annotations

import argparse

from app.core.settings import get_settings
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.vector_store import (
    create_qdrant_client,
    search_dense_chunks,
)


def parse_arguments() -> argparse.Namespace:
    """Parse dense-search arguments."""

    settings = get_settings()

    parser = argparse.ArgumentParser(description="Search OTSentinel chunks using dense retrieval.")

    parser.add_argument(
        "query",
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--config-name",
        default="section-bge-m3-512-64",
    )

    parser.add_argument(
        "--source-id",
        default="nist-sp-800-82-r3",
    )

    parser.add_argument(
        "--model-name",
        default=settings.embedding_model_name,
    )

    parser.add_argument(
        "--device",
        default=settings.embedding_device,
    )

    parser.add_argument(
        "--collection-name",
        default=settings.qdrant_collection,
    )

    return parser.parse_args()


def main() -> None:
    """Embed a query and retrieve similar chunks."""

    arguments = parse_arguments()
    settings = get_settings()

    embedder = DenseEmbedder(
        model_name=arguments.model_name,
        requested_device=arguments.device,
    )

    query_vector = embedder.encode_query(arguments.query)

    client = create_qdrant_client(settings.qdrant_url)

    results = search_dense_chunks(
        client=client,
        collection_name=arguments.collection_name,
        query_vector=query_vector,
        top_k=arguments.top_k,
        config_name=arguments.config_name,
        source_id=arguments.source_id,
    )

    print(f"\nQuery: {arguments.query}")
    print(f"Results: {len(results)}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        payload = result.payload or {}
        text = str(payload.get("text", ""))

        preview = text[:500].replace(
            "\n",
            " ",
        )

        print(
            f"[{rank}] Score: {result.score:.4f}\n"
            f"Heading: {payload.get('heading')}\n"
            f"Pages: {payload.get('page_start')}-"
            f"{payload.get('page_end')}\n"
            f"Chunk: {payload.get('chunk_id')}\n"
            f"Text: {preview}\n"
        )


if __name__ == "__main__":
    main()
