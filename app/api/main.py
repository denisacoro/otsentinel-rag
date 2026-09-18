"""FastAPI service entry point (steps 15-16).

Wraps app.generation.rag_pipeline.answer_question() and
app.ingestion.upload_pipeline.ingest_uploaded_pdf() behind an HTTP API. The
embedding model, sparse embedder, tokenizer, and Qdrant client are created
once at startup (via the lifespan handler below) and stored on app.state,
then injected into every /ask and /upload call. Without this, every request
would reload the ~2GB BGE-M3 embedding model from scratch.

Run locally:
    uvicorn app.api.main:app --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
         -d "{\"question\": \"What is a physical access control system?\"}"

Or use the interactive docs at http://localhost:8000/docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from app.api.routes import router
from app.core.settings import get_settings
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse_embedder import SparseEmbedder
from app.retrieval.vector_store import create_qdrant_client

TOKENIZER_NAME = "BAAI/bge-m3"  # must match scripts/build_chunks.py's --tokenizer-name default


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    print("Loading embedding model, tokenizer, and connecting to Qdrant (once, at startup)...")
    app.state.settings = settings
    app.state.dense_embedder = DenseEmbedder(
        model_name=settings.embedding_model_name,
        requested_device=settings.embedding_device,
    )
    app.state.sparse_embedder = SparseEmbedder()
    app.state.qdrant_client = create_qdrant_client(settings.qdrant_url)
    app.state.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    print("Startup complete.")

    yield

    # Nothing needs explicit async cleanup here -- the embedders, tokenizer,
    # and Qdrant client don't hold resources that require a shutdown call.


app = FastAPI(
    title="OTSentinel AI",
    description="Multilingual RAG for CPS/SCADA/OT security documentation.",
    version="0.2.0",
    lifespan=lifespan,
)

# Permissive CORS is a deliberate simplification for a local, single-user
# portfolio demo (the Gradio UI calls this from a different port). Not
# appropriate if this were ever exposed beyond localhost -- revisit in step 18
# (testing/security) if that changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"service": "OTSentinel AI", "docs": "/docs", "health": "/health"}
