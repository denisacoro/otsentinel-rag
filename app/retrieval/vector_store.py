from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.schemas.document import DocumentChunk

DENSE_VECTOR_NAME = "dense"


def create_qdrant_client(
    qdrant_url: str,
) -> QdrantClient:
    """Create a Qdrant server client."""

    return QdrantClient(
        url=qdrant_url,
        timeout=60,
    )


def chunk_point_id(chunk_id: str) -> str:
    """Create a stable Qdrant UUID from a chunk ID."""

    return str(
        uuid5(
            NAMESPACE_URL,
            f"otsentinel-ai:{chunk_id}",
        )
    )


def ensure_dense_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    """Create the dense-vector collection when necessary."""

    collection_exists = client.collection_exists(collection_name=collection_name)

    if collection_exists and recreate:
        client.delete_collection(collection_name=collection_name)
        collection_exists = False

    if collection_exists:
        print(f"Using existing Qdrant collection: {collection_name}")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
    )

    print(f"Created Qdrant collection: {collection_name}\nDense vector size: {vector_size}")


def create_payload_indexes(
    *,
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Create indexes for fields used in retrieval filters."""

    keyword_fields = [
        "source_id",
        "document_id",
        "section_id",
        "config_name",
        "language",
        "publisher",
        "document_type",
    ]

    for field_name in keyword_fields:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    for field_name in ["page_start", "page_end"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True,
        )


def create_chunk_payload(
    *,
    chunk: DocumentChunk,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create searchable metadata stored with a Qdrant point."""

    return {
        "source_id": chunk.source_id,
        "document_id": chunk.document_id,
        "section_id": chunk.section_id,
        "chunk_id": chunk.chunk_id,
        "config_name": chunk.config_name,
        "title": chunk.title,
        "heading": chunk.heading,
        "heading_level": chunk.heading_level,
        "heading_path": chunk.heading_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "chunk_count": chunk.chunk_count,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "content_sha256": chunk.content_sha256,
        "embedding_model": source_metadata["embedding_model"],
        "language": source_metadata["language"],
        "publisher": source_metadata["publisher"],
        "document_type": source_metadata["document_type"],
        "version": source_metadata["version"],
        "published_at": source_metadata["published_at"],
    }


def upsert_dense_chunks(
    *,
    client: QdrantClient,
    collection_name: str,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[Sequence[float]],
    source_metadata: dict[str, Any],
) -> None:
    """Insert or replace one batch of embedded chunks."""

    if len(chunks) != len(vectors):
        raise ValueError("The chunk count must equal the vector count.")

    points: list[models.PointStruct] = []

    for chunk, vector in zip(
        chunks,
        vectors,
        strict=True,
    ):
        points.append(
            models.PointStruct(
                id=chunk_point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR_NAME: list(vector),
                },
                payload=create_chunk_payload(
                    chunk=chunk,
                    source_metadata=source_metadata,
                ),
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )


def build_search_filter(
    *,
    config_name: str | None = None,
    source_id: str | None = None,
) -> models.Filter | None:
    """Build optional metadata filters for dense retrieval."""

    conditions: list[models.FieldCondition] = []

    if config_name is not None:
        conditions.append(
            models.FieldCondition(
                key="config_name",
                match=models.MatchValue(value=config_name),
            )
        )

    if source_id is not None:
        conditions.append(
            models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def search_dense_chunks(
    *,
    client: QdrantClient,
    collection_name: str,
    query_vector: Sequence[float],
    top_k: int,
    config_name: str | None = None,
    source_id: str | None = None,
) -> list[models.ScoredPoint]:
    """Search indexed chunks by dense-vector similarity."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_filter = build_search_filter(
        config_name=config_name,
        source_id=source_id,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return response.points
