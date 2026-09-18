"""FastAPI service entry point (step 15).

Wraps app.generation.rag_pipeline.answer_question() behind an HTTP API. The
embedding model, sparse embedder, and Qdrant client are created once at
startup (via the lifespan handler below) and stored on app.state, then
injected into every /ask call -- see rag_pipeline.answer_question()'s
dense_embedder/sparse_embedder/qdrant_client params. Without this, every
request would reload the ~2GB BGE-M3 embedding model from scratch.

Run locally:
    uvicorn app.api.main:app --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
         -d "{\"question\": \"What is a physical access control system?\"}"
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.settings import get_settings
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse_embedder import SparseEmbedder
from app.retrieval.vector_store import create_qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    print("Loading embedding model and connecting to Qdrant (once, at startup)...")
    app.state.settings = settings
    app.state.dense_embedder = DenseEmbedder(
        model_name=settings.embedding_model_name,
        requested_device=settings.embedding_device,
    )
    app.state.sparse_embedder = SparseEmbedder()
    app.state.qdrant_client = create_qdrant_client(settings.qdrant_url)
    print("Startup complete.")

    yield

    # Nothing needs explicit async cleanup here -- the embedders and Qdrant
    # client don't hold resources that require a shutdown call.


app = FastAPI(
    title="OTSentinel AI",
    description="Multilingual RAG for CPS/SCADA/OT security documentation.",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS is a deliberate simplification for a local, single-user
# portfolio demo (step 16's Gradio UI will call this). Not appropriate if this
# were ever exposed beyond localhost -- revisit in step 18 (testing/security)
# if that changes.
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
