from __future__ import annotations

import argparse
from pathlib import Path

from app.core.settings import get_settings
from app.ingestion.downloader import load_manifest
from app.retrieval.chunk_loader import load_chunks_jsonl
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.vector_store import (
    create_payload_indexes,
    create_qdrant_client,
    ensure_dense_collection,
    upsert_dense_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sources.jsonl"

CHUNKS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "chunks"


def parse_arguments() -> argparse.Namespace:
    """Parse indexing command-line arguments."""

    settings = get_settings()

    parser = argparse.ArgumentParser(
        description=("Generate dense BGE-M3 embeddings and index chunks in Qdrant.")
    )

    parser.add_argument(
        "--source-id",
        default="nist-sp-800-82-r3",
    )

    parser.add_argument(
        "--config-name",
        default="section-bge-m3-512-64",
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
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
    )

    parser.add_argument(
        "--collection-name",
        default=settings.qdrant_collection,
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Index only the first N chunks for a smoke test.",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection.",
    )

    return parser.parse_args()


def batch_items(
    items: list,
    batch_size: int,
):
    """Yield successive list batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start_index in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[start_index : start_index + batch_size]


def main() -> None:
    """Embed and index one chunk configuration."""

    arguments = parse_arguments()
    settings = get_settings()

    sources = load_manifest(MANIFEST_PATH)

    sources_by_id = {source.source_id: source for source in sources}

    if arguments.source_id not in sources_by_id:
        raise ValueError(f"Unknown source ID: {arguments.source_id}")

    source = sources_by_id[arguments.source_id]

    chunks_path = CHUNKS_DIRECTORY / (f"{source.source_id}.{arguments.config_name}.chunks.jsonl")

    print(f"Loading chunks: {chunks_path}")

    chunks = load_chunks_jsonl(chunks_path)

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("--limit must be greater than zero.")

        chunks = chunks[: arguments.limit]

    print(f"Chunks to index: {len(chunks)}")

    embedder = DenseEmbedder(
        model_name=arguments.model_name,
        requested_device=arguments.device,
    )

    print(f"Embedding dimension: {embedder.embedding_dimension}")

    client = create_qdrant_client(settings.qdrant_url)

    # Confirm that the Qdrant server is reachable.
    client.get_collections()

    ensure_dense_collection(
        client=client,
        collection_name=arguments.collection_name,
        vector_size=embedder.embedding_dimension,
        recreate=arguments.recreate,
    )

    create_payload_indexes(
        client=client,
        collection_name=arguments.collection_name,
    )

    source_metadata = {
        "embedding_model": arguments.model_name,
        "language": source.language,
        "publisher": source.publisher,
        "document_type": source.document_type,
        "version": source.version,
        "published_at": source.published_at,
    }

    indexed_count = 0

    for batch_number, chunk_batch in enumerate(
        batch_items(
            chunks,
            arguments.batch_size,
        ),
        start=1,
    ):
        embedding_texts = [chunk.embedding_text for chunk in chunk_batch]

        vectors = embedder.encode_documents(
            embedding_texts,
            batch_size=arguments.batch_size,
            show_progress=False,
        )

        upsert_dense_chunks(
            client=client,
            collection_name=arguments.collection_name,
            chunks=chunk_batch,
            vectors=vectors.tolist(),
            source_metadata=source_metadata,
        )

        indexed_count += len(chunk_batch)

        print(f"Batch {batch_number}: {indexed_count}/{len(chunks)} indexed")

    collection_info = client.get_collection(collection_name=arguments.collection_name)

    print(
        "\nIndexing completed.\n"
        f"  Collection: {arguments.collection_name}\n"
        f"  Source: {source.source_id}\n"
        f"  Configuration: {arguments.config_name}\n"
        f"  Model: {arguments.model_name}\n"
        f"  Device: {embedder.device}\n"
        f"  Indexed this run: {indexed_count}\n"
        f"  Qdrant points: {collection_info.points_count}"
    )


if __name__ == "__main__":
    main()
