"""Pydantic request/response models for the HTTP API layer.

These are distinct from app/schemas/* (the internal pipeline's data
contracts, e.g. RagAnswer, DocumentChunk) -- this module only defines what the
API accepts as input. The response type for /ask is app.schemas.answer.RagAnswer
directly, reused as-is rather than duplicated, since it's already a complete,
well-formed Pydantic model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    language: str = Field("en", pattern="^(en|ro)$", description="'en' or 'ro'.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    source_id: str | None = Field(
        None, description="Optional: restrict retrieval to a single source document."
    )


class HealthResponse(BaseModel):
    status: str  # "ok" or "degraded"
    qdrant_ok: bool
    ollama_ok: bool
    embedding_model_loaded: bool
