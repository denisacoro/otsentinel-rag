"""Step 15: the /ask and /health endpoints.

Only the base model is served here (Ollama, production default). The
fine-tuned adapter is not deployed (see docs/FINETUNING_RESULTS.md) and stays
reachable only through scripts/ask_with_model.py and the evaluation scripts --
loading it here would pull unsloth/torch into the API process and try to fit
it in VRAM alongside whatever else is running, for a model that showed no
measured improvement over baseline.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AskRequest, HealthResponse
from app.generation.rag_pipeline import answer_question
from app.schemas.answer import RagAnswer

router = APIRouter()


@router.post("/ask", response_model=RagAnswer)
def ask(payload: AskRequest, request: Request) -> RagAnswer:
    try:
        return answer_question(
            question=payload.question,
            language=payload.language,
            top_k=payload.top_k,
            source_id=payload.source_id,
            dense_embedder=request.app.state.dense_embedder,
            sparse_embedder=request.app.state.sparse_embedder,
            qdrant_client=request.app.state.qdrant_client,
        )
    except Exception as exc:
        # Broad on purpose for v1 -- Qdrant unreachable, Ollama unreachable, or
        # an unexpected pipeline error all surface as a 503 with the real
        # cause in the detail rather than an opaque 500 stack trace. Finer-
        # grained error handling belongs to step 18 (testing/security), not
        # this first working version.
        raise HTTPException(status_code=503, detail=f"RAG pipeline failed: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings

    qdrant_ok = False
    try:
        request.app.state.qdrant_client.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    ollama_ok = False
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    embedding_model_loaded = request.app.state.dense_embedder is not None

    status = "ok" if (qdrant_ok and ollama_ok and embedding_model_loaded) else "degraded"

    return HealthResponse(
        status=status,
        qdrant_ok=qdrant_ok,
        ollama_ok=ollama_ok,
        embedding_model_loaded=embedding_model_loaded,
    )
