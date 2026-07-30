from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunkRef(BaseModel):
    """One retrieved chunk shown alongside a generated answer."""

    chunk_id: str
    source_id: str
    score: float
    heading: str | None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class RagAnswer(BaseModel):
    """The full result of one baseline RAG query."""

    question: str
    language: Literal["en", "ro"]
    source_id: str | None
    config_name: str
    model_name: str
    top_k: int = Field(ge=1)

    retrieved_chunks: list[RetrievedChunkRef]
    answer_text: str
    cited_chunk_ids: list[str]
    refused: bool

    retrieval_latency_ms: float = Field(ge=0)
    generation_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)

    queried_at_utc: datetime