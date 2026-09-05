from __future__ import annotations

import re
from datetime import UTC, datetime

from app.core.settings import get_settings
from app.generation.llm_client import OllamaClient
from app.generation.prompts import REFUSAL_TEXT, build_messages
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse_embedder import SparseEmbedder
from app.retrieval.vector_store import create_qdrant_client, search_hybrid_chunks
from app.schemas.answer import RagAnswer, RetrievedChunkRef
from app.schemas.document import DocumentChunk

CITATION_TAG_PATTERN = re.compile(r"\[S(\d+)\]")


def extract_cited_chunk_ids(answer_text: str, tag_to_chunk_id: dict[str, str]) -> list[str]:
    """Resolve cited [S1]-style tags back to real chunk IDs, in order of first use."""

    cited: list[str] = []

    for number in CITATION_TAG_PATTERN.findall(answer_text):
        tag = f"S{number}"
        chunk_id = tag_to_chunk_id.get(tag)

        if chunk_id and chunk_id not in cited:
            cited.append(chunk_id)

    return cited


def _payload_to_chunk(payload: dict) -> DocumentChunk:
    """Rebuild a DocumentChunk from a Qdrant payload for prompt building."""

    return DocumentChunk(
        source_id=payload["source_id"],
        document_id=payload["document_id"],
        section_id=payload["section_id"],
        chunk_id=payload["chunk_id"],
        config_name=payload["config_name"],
        title=payload["title"],
        heading=payload.get("heading"),
        heading_level=payload.get("heading_level", 0),
        heading_path=payload.get("heading_path", []),
        page_start=payload["page_start"],
        page_end=payload["page_end"],
        chunk_index=payload["chunk_index"],
        chunk_count=payload["chunk_count"],
        text=payload["text"],
        embedding_text=payload["text"],
        body_token_count=payload["token_count"],
        token_count=payload["token_count"],
        token_start=0,
        token_end=payload["token_count"],
        overlap_with_previous=0,
        tokenizer_name="unknown",
        content_sha256=payload["content_sha256"],
    )


def answer_question(
    *,
    question: str,
    source_id: str | None = None,
    language: str = "en",
    top_k: int = 5,
    config_name: str = "section-bge-m3-512-64",
    generator: object | None = None,
) -> RagAnswer:
    """Run the hybrid (dense + sparse, RRF-fused) RAG pipeline for one question."""

    settings = get_settings()
    generator_label = getattr(generator, "model_name", None) or settings.generator_model_name
    total_start = datetime.now(UTC)

    dense_embedder = DenseEmbedder(
        model_name=settings.embedding_model_name,
        requested_device=settings.embedding_device,
    )
    sparse_embedder = SparseEmbedder()

    dense_query_vector = dense_embedder.encode_query(question)
    sparse_query_vector = sparse_embedder.encode_query(question)

    client = create_qdrant_client(settings.qdrant_url)

    retrieval_start = datetime.now(UTC)
    results = search_hybrid_chunks(
        client=client,
        collection_name=settings.qdrant_collection,
        dense_query_vector=dense_query_vector,
        sparse_query_vector=sparse_query_vector,
        top_k=top_k,
        prefetch_limit=settings.hybrid_prefetch_limit,
        config_name=config_name,
        source_id=source_id,
    )
    retrieval_latency_ms = (datetime.now(UTC) - retrieval_start).total_seconds() * 1000

    retrieved_refs = [
        RetrievedChunkRef(
            chunk_id=str((r.payload or {}).get("chunk_id")),
            source_id=str((r.payload or {}).get("source_id")),
            score=r.score,
            heading=(r.payload or {}).get("heading"),
            page_start=(r.payload or {}).get("page_start", 1),
            page_end=(r.payload or {}).get("page_end", 1),
        )
        for r in results
    ]

    if not results or results[0].score < settings.min_retrieval_score:
        queried_at = datetime.now(UTC)
        total_latency_ms = (queried_at - total_start).total_seconds() * 1000

        return RagAnswer(
            question=question,
            language=language,
            source_id=source_id,
            config_name=config_name,
            model_name=generator_label,
            top_k=top_k,
            retrieved_chunks=retrieved_refs,
            answer_text=REFUSAL_TEXT,
            cited_chunk_ids=[],
            refused=True,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=0.0,
            total_latency_ms=total_latency_ms,
            queried_at_utc=queried_at,
        )

    chunks = [_payload_to_chunk(r.payload or {}) for r in results]
    messages, tag_to_chunk_id = build_messages(question=question, language=language, chunks=chunks)

    llm = generator if generator is not None else OllamaClient(
        base_url=settings.ollama_base_url,
        model_name=settings.generator_model_name,
        temperature=settings.generation_temperature,
    )
    answer_text, generation_latency_ms = llm.chat(messages)

    cited_chunk_ids = extract_cited_chunk_ids(answer_text, tag_to_chunk_id)
    refused = REFUSAL_TEXT in answer_text

    queried_at = datetime.now(UTC)
    total_latency_ms = (queried_at - total_start).total_seconds() * 1000

    return RagAnswer(
        question=question,
        language=language,
        source_id=source_id,
        config_name=config_name,
        model_name=generator_label,
        top_k=top_k,
        retrieved_chunks=retrieved_refs,
        answer_text=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        refused=refused,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
        queried_at_utc=queried_at,
    )