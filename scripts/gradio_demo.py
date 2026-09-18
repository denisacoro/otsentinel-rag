"""Step 16: Gradio demo UI.

Pure HTTP client -- no models loaded in this process. Everything goes through
the FastAPI service from step 15/16 (app/api/main.py), which owns the one
loaded copy of the embedding model, Qdrant connection, and Ollama client.
Running the embedding model a second time here, in a separate process, would
mean two ~2GB models competing for the same 4GB GPU.

Two tabs:
  - "Ask the corpus": chat over the fixed corpus (NIST SP 800-82r3, MQTT v5.0).
  - "Upload your own document": upload a PDF, ingest it via POST /upload
    (chunked, embedded, and indexed into the same Qdrant collection under a
    unique source_id), then ask questions scoped to just that document via
    source_id filtering on the existing /ask endpoint. A delete button removes
    it from Qdrant afterward (DELETE /documents/{source_id} only ever accepts
    an uploaded document's source_id, never the fixed corpus's).

Requires the FastAPI service to already be running:
    uvicorn app.api.main:app --port 8000

Then, in a second terminal:
    python scripts/gradio_demo.py

Opens at http://localhost:7860.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import httpx

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_S = 120.0
UPLOAD_TIMEOUT_S = 300.0  # parsing + chunking + embedding a whole PDF can take a while


def _format_sources_and_latency(data: dict) -> str:
    lines = []
    for chunk in data.get("retrieved_chunks", []):
        heading = chunk.get("heading") or "(no heading)"
        lines.append(
            f"- {heading}  (p. {chunk.get('page_start')}-{chunk.get('page_end')}, "
            f"score {chunk.get('score', 0):.3f})"
        )
    sources = "\n".join(lines) if lines else "(no chunks retrieved)"

    latency = (
        f"\nRetrieval: {data.get('retrieval_latency_ms', 0):.0f}ms   "
        f"Generation: {data.get('generation_latency_ms', 0):.0f}ms   "
        f"Total: {data.get('total_latency_ms', 0):.0f}ms"
    )
    refused = "\n(This was a retrieval-layer refusal -- no evidence met the confidence threshold.)" if data.get("refused") else ""
    return sources + latency + refused


def _ask(question: str, language: str, top_k: int, source_id: str | None) -> tuple[str, str, str]:
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    payload = {"question": question, "language": language, "top_k": int(top_k)}
    if source_id:
        payload["source_id"] = source_id

    try:
        response = httpx.post(f"{API_BASE_URL}/ask", json=payload, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        return f"API error ({exc.response.status_code}): {exc.response.text}", "", ""
    except Exception as exc:
        return f"Could not reach the API at {API_BASE_URL} -- is `uvicorn app.api.main:app --port 8000` running? ({exc})", "", ""

    answer = data.get("answer_text", "")
    citations = ", ".join(data.get("cited_chunk_ids", [])) or "(none)"
    details = _format_sources_and_latency(data)
    return answer, citations, details


def ask_corpus(question: str, language: str, top_k: int) -> tuple[str, str, str]:
    return _ask(question, language, top_k, source_id=None)


def ask_own_document(question: str, language: str, top_k: int, source_id: str | None) -> tuple[str, str, str]:
    if not source_id:
        return "Upload a document first.", "", ""
    return _ask(question, language, top_k, source_id=source_id)


def _resolve_upload_path(file_obj) -> Path:
    if file_obj is None:
        raise ValueError("No file provided.")
    if isinstance(file_obj, str):
        return Path(file_obj)
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    raise ValueError(f"Unrecognized file input type: {type(file_obj)}")


def upload_document(file_obj, title: str, language: str):
    if file_obj is None:
        return "Please choose a PDF file first.", None, gr.update(visible=False)

    try:
        path = _resolve_upload_path(file_obj)
    except ValueError as exc:
        return str(exc), None, gr.update(visible=False)

    try:
        with path.open("rb") as f:
            files = {"file": (path.name, f, "application/pdf")}
            data = {"language": language}
            if title and title.strip():
                data["title"] = title.strip()
            response = httpx.post(f"{API_BASE_URL}/upload", files=files, data=data, timeout=UPLOAD_TIMEOUT_S)
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as exc:
        return f"Upload failed ({exc.response.status_code}): {exc.response.text}", None, gr.update(visible=False)
    except Exception as exc:
        return f"Could not reach the API at {API_BASE_URL} -- is `uvicorn app.api.main:app --port 8000` running? ({exc})", None, gr.update(visible=False)

    summary = (
        f"Ingested \"{result['title']}\" -- {result['total_pages']} pages, "
        f"{result['num_sections']} sections, {result['num_chunks']} chunks.\n"
        f"Source ID: {result['source_id']}\n"
        "You can now ask questions scoped to just this document below."
    )
    return summary, result["source_id"], gr.update(visible=True)


def delete_document(source_id: str | None) -> str:
    if not source_id:
        return "No document to delete."
    try:
        response = httpx.delete(f"{API_BASE_URL}/documents/{source_id}", timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"Delete failed ({exc.response.status_code}): {exc.response.text}"
    except Exception as exc:
        return f"Could not reach the API: {exc}"
    return f"Deleted {source_id}. Upload a new document to continue, or refresh the page to reset this tab."


with gr.Blocks(title="OTSentinel AI") as demo:
    gr.Markdown("# OTSentinel AI\nMultilingual RAG for CPS/SCADA/OT security documentation.")

    with gr.Tab("Ask the corpus"):
        gr.Markdown("Ask questions against the fixed corpus: NIST SP 800-82r3 and MQTT v5.0.")

        question_box = gr.Textbox(
            label="Question", placeholder="e.g. What is a physical access control system?", lines=2
        )
        with gr.Row():
            language_dd = gr.Radio(["en", "ro"], value="en", label="Language")
            top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Chunks to retrieve (top_k)")
        ask_btn = gr.Button("Ask", variant="primary")

        answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
        with gr.Accordion("Citations & retrieved sources", open=False):
            citations_box = gr.Textbox(label="Cited chunk IDs", interactive=False)
            sources_box = gr.Textbox(label="Retrieved sources / latency", lines=8, interactive=False)

        ask_btn.click(
            ask_corpus,
            inputs=[question_box, language_dd, top_k_slider],
            outputs=[answer_box, citations_box, sources_box],
        )

    with gr.Tab("Upload your own document"):
        gr.Markdown(
            "Upload a PDF (e.g. a scientific paper or report) and ask questions "
            "scoped to just that document -- same retrieval/generation pipeline "
            "as the main corpus, filtered to your upload. Documents with no "
            "extractable text (scanned/image-only PDFs) will be rejected; "
            "there's no OCR step."
        )
        uploaded_source_id = gr.State(None)

        file_input = gr.File(label="PDF file", file_types=[".pdf"])
        with gr.Row():
            title_input = gr.Textbox(label="Title (optional)", placeholder="Defaults to the filename")
            upload_language_dd = gr.Radio(["en", "ro"], value="en", label="Document language")
        upload_btn = gr.Button("Ingest document", variant="primary")
        upload_status = gr.Textbox(label="Status", lines=4, interactive=False)

        with gr.Group(visible=False) as ask_own_group:
            own_question_box = gr.Textbox(label="Question about your document", lines=2)
            own_top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Chunks to retrieve (top_k)")
            own_ask_btn = gr.Button("Ask")
            own_answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
            with gr.Accordion("Citations & retrieved sources", open=False):
                own_citations_box = gr.Textbox(label="Cited chunk IDs", interactive=False)
                own_sources_box = gr.Textbox(label="Retrieved sources / latency", lines=8, interactive=False)
            delete_btn = gr.Button("Delete this document", variant="stop")
            delete_status = gr.Textbox(show_label=False, interactive=False)

        upload_btn.click(
            upload_document,
            inputs=[file_input, title_input, upload_language_dd],
            outputs=[upload_status, uploaded_source_id, ask_own_group],
        )
        own_ask_btn.click(
            ask_own_document,
            inputs=[own_question_box, upload_language_dd, own_top_k_slider, uploaded_source_id],
            outputs=[own_answer_box, own_citations_box, own_sources_box],
        )
        delete_btn.click(delete_document, inputs=[uploaded_source_id], outputs=[delete_status])


if __name__ == "__main__":
    demo.queue().launch(server_port=7860)
