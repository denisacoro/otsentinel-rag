"""Ingest one user-uploaded PDF into the same Qdrant collection used by the
production corpus, tagged with a unique source_id so it can be queried in
isolation via the source_id filtering the retrieval/generation pipeline
already supports. Mirrors scripts/build_sections.py -> build_chunks.py ->
index_chunks.py, but runs entirely in-memory for a single document processed
synchronously on upload, rather than a batch corpus build with intermediate
JSONL files on disk.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingestion.chunker import build_document_chunks
from app.ingestion.structure_extractor import extract_document_structure
from app.retrieval.vector_store import upsert_hybrid_chunks

DEFAULT_CONFIG_NAME = "section-bge-m3-512-64"
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_BATCH_SIZE = 16

UPLOAD_SOURCE_ID_PREFIX = "upload-"


def make_upload_source_id(filename: str) -> str:
    """Generate a unique, Qdrant/filesystem-safe source_id for an uploaded
    file. Always starts with UPLOAD_SOURCE_ID_PREFIX so uploaded documents can
    be distinguished from -- and only ever delete -- the fixed corpus."""

    stem = Path(filename).stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "document"
    suffix = uuid.uuid4().hex[:8]
    return f"{UPLOAD_SOURCE_ID_PREFIX}{slug}-{suffix}"


def ingest_uploaded_pdf(
    *,
    pdf_path: Path,
    title: str,
    language: str,
    dense_embedder: Any,
    sparse_embedder: Any,
    qdrant_client: Any,
    tokenizer: Any,
    collection_name: str,
    embedding_model_name: str,
    tokenizer_name: str = "BAAI/bge-m3",
    config_name: str = DEFAULT_CONFIG_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Parse, chunk, embed and index one uploaded PDF.

    dense_embedder / sparse_embedder / qdrant_client / tokenizer are expected
    to be already-loaded, long-lived instances (the FastAPI service's
    app.state) -- this function never loads its own models, to avoid a second
    copy of the ~2GB embedding model competing for VRAM alongside whatever
    else is running.

    Returns a small summary dict (source_id, title, num_sections, num_chunks,
    total_pages) the caller can show the user. Raises ValueError for a PDF
    that produces no usable text (e.g. scanned/image-only -- this pipeline has
    no OCR step) so the caller can turn that into a clear 422, not a 500.
    """

    source_id = make_upload_source_id(pdf_path.name)

    sections, structure_summary, _lines = extract_document_structure(
        source_id=source_id,
        document_id=source_id,
        title=title,
        pdf_path=pdf_path,
    )

    if not sections:
        raise ValueError(
            "No extractable text sections were found in this PDF. It may be a "
            "scanned or image-only document -- this pipeline has no OCR step."
        )

    chunks, _chunk_summary = build_document_chunks(
        source_id=source_id,
        document_id=source_id,
        title=title,
        sections=sections,
        sections_path=pdf_path,  # metadata only -- chunker.py never reads this back from disk
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        config_name=config_name,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunks:
        raise ValueError("Parsing succeeded but produced zero chunks -- the PDF may be empty.")

    source_metadata = {
        "embedding_model": embedding_model_name,
        "language": language,
        "publisher": "user-upload",
        "document_type": "user-uploaded",
        "version": "1",
        "published_at": datetime.now(UTC).date().isoformat(),
    }

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embedding_texts = [chunk.embedding_text for chunk in batch]

        dense_vectors = dense_embedder.encode_documents(
            embedding_texts, batch_size=batch_size, show_progress=False
        )
        sparse_vectors = sparse_embedder.encode_documents(embedding_texts)

        upsert_hybrid_chunks(
            client=qdrant_client,
            collection_name=collection_name,
            chunks=batch,
            dense_vectors=dense_vectors.tolist(),
            sparse_vectors=sparse_vectors,
            source_metadata=source_metadata,
        )

    return {
        "source_id": source_id,
        "title": title,
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "total_pages": structure_summary.total_pages,
    }


def delete_uploaded_document(*, qdrant_client: Any, collection_name: str, source_id: str) -> None:
    """Remove every chunk for one uploaded document from Qdrant.

    Caller is responsible for only invoking this with an uploaded document's
    source_id (see UPLOAD_SOURCE_ID_PREFIX) -- this function itself performs
    no such check, since it operates purely on whatever source_id it's given.
    """

    from qdrant_client import models

    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))]
            )
        ),
    )
