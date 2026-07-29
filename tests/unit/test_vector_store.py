from qdrant_client import QdrantClient

from app.retrieval.vector_store import (
    chunk_point_id,
    ensure_dense_collection,
    search_dense_chunks,
    upsert_dense_chunks,
)
from app.schemas.document import DocumentChunk


def create_test_chunk(
    *,
    chunk_number: int,
    text: str,
) -> DocumentChunk:
    """Create one valid retrieval chunk for testing."""

    return DocumentChunk(
        source_id="test-source",
        document_id="test-document",
        section_id="test-section",
        chunk_id=f"test-chunk-{chunk_number}",
        config_name="test-config",
        title="Test OT Document",
        heading="Network Segmentation",
        heading_level=2,
        heading_path=[
            "OT Security",
            "Network Segmentation",
        ],
        page_start=10,
        page_end=11,
        chunk_index=chunk_number,
        chunk_count=2,
        text=text,
        embedding_text=text,
        body_token_count=4,
        token_count=4,
        token_start=0,
        token_end=4,
        overlap_with_previous=0,
        tokenizer_name="test-tokenizer",
        content_sha256="a" * 64,
    )


def test_chunk_point_id_is_deterministic() -> None:
    first_id = chunk_point_id("same-chunk")
    second_id = chunk_point_id("same-chunk")

    assert first_id == second_id
    assert chunk_point_id("different-chunk") != first_id


def test_upserts_and_searches_dense_chunks() -> None:
    client = QdrantClient(":memory:")

    collection_name = "test-collection"

    ensure_dense_collection(
        client=client,
        collection_name=collection_name,
        vector_size=3,
    )

    security_chunk = create_test_chunk(
        chunk_number=1,
        text="Industrial network segmentation guidance.",
    )

    unrelated_chunk = create_test_chunk(
        chunk_number=2,
        text="Unrelated environmental information.",
    )

    source_metadata = {
        "embedding_model": "test-model",
        "language": "en",
        "publisher": "Test Publisher",
        "document_type": "test",
        "version": "1",
        "published_at": "2026",
    }

    upsert_dense_chunks(
        client=client,
        collection_name=collection_name,
        chunks=[
            security_chunk,
            unrelated_chunk,
        ],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        source_metadata=source_metadata,
    )

    results = search_dense_chunks(
        client=client,
        collection_name=collection_name,
        query_vector=[1.0, 0.0, 0.0],
        top_k=1,
        config_name="test-config",
        source_id="test-source",
    )

    assert len(results) == 1

    payload = results[0].payload

    assert payload is not None
    assert payload["chunk_id"] == "test-chunk-1"
    assert "network segmentation" in payload["text"].lower()
