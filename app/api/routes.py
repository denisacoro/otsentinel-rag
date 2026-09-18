"""Step 15/16: /ask, /health, /upload, and /documents/{source_id}.

Only the base model is served here (Ollama, production default). The
fine-tuned adapter is not deployed (see docs/FINETUNING_RESULTS.md) and stays
reachable only through scripts/ask_with_model.py and the evaluation scripts --
loading it here would pull unsloth/torch into the API process and try to fit
it in VRAM alongside whatever else is running, for a model that showed no
measured improvement over baseline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import AskRequest, DeleteResponse, HealthResponse, UploadResponse
from app.generation.rag_pipeline import answer_question
from app.ingestion.upload_pipeline import (
    UPLOAD_SOURCE_ID_PREFIX,
    delete_uploaded_document,
    ingest_uploaded_pdf,
)
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


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    language: str = Form("en"),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = request.app.state.settings

    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = ingest_uploaded_pdf(
            pdf_path=tmp_path,
            title=title or Path(file.filename).stem,
            language=language,
            dense_embedder=request.app.state.dense_embedder,
            sparse_embedder=request.app.state.sparse_embedder,
            qdrant_client=request.app.state.qdrant_client,
            tokenizer=request.app.state.tokenizer,
            collection_name=settings.qdrant_collection,
            embedding_model_name=settings.embedding_model_name,
        )
    except ValueError as exc:
        # A clean, expected failure -- e.g. a scanned PDF with no extractable
        # text. 422 (not 500): the request was well-formed, the file itself
        # couldn't be processed.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(**result)


@router.delete("/documents/{source_id}", response_model=DeleteResponse)
def delete_document(source_id: str, request: Request) -> DeleteResponse:
    if not source_id.startswith(UPLOAD_SOURCE_ID_PREFIX):
        # Guards the fixed corpus (nist-sp-800-82-r3, mqtt-v5-0) -- this
        # endpoint can only ever remove documents that came in through
        # /upload, never the production sources.
        raise HTTPException(
            status_code=400,
            detail=f"Only uploaded documents (source_id starting with '{UPLOAD_SOURCE_ID_PREFIX}') can be deleted through this endpoint.",
        )

    settings = request.app.state.settings
    delete_uploaded_document(
        qdrant_client=request.app.state.qdrant_client,
        collection_name=settings.qdrant_collection,
        source_id=source_id,
    )
    return DeleteResponse(deleted_source_id=source_id)
