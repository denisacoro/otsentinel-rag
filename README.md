# OTSentinel AI

Multilingual RAG and QLoRA fine-tuned LLM platform for CPS, SCADA and Industrial IoT security.
Answers English and Romanian questions using evidence retrieved from authoritative OT/ICS
documentation, with source citations and explicit refusal when evidence is insufficient.

See [`docs/project_specification.md`](docs/project_specification.md) for scope, personas, and
success criteria.

## Current status

Ingestion, parsing and indexing pipeline is working end to end for a single-source corpus:

- Source manifest + checksummed, idempotent downloader (`app/ingestion/downloader.py`)
- PDF parsing with header/footer removal and heading detection (`app/ingestion/pdf_parser.py`,
  `app/ingestion/structure_extractor.py`)
- Structure-aware, token-window chunking with overlap (`app/ingestion/chunker.py`)
- Dense embeddings (BGE-M3) and Qdrant indexing with metadata filters (`app/retrieval/`)
- CLI search over indexed chunks (`scripts/search_chunks.py`)

Corpus so far: NIST SP 800-82r3, MQTT 5.0.

Not yet built: hybrid/reranked retrieval, LLM generation, evaluation, fine-tuning, API beyond
`/health`, Gradio demo, Docker Compose for the full stack, CI. See
`docs/project_specification.md` and the project roadmap for the full plan.

## Running the pipeline locally

```bash
python scripts/download_sources.py
python scripts/parse_documents.py
python scripts/build_sections.py
python scripts/build_chunks.py
python scripts/index_chunks.py --source-id <source-id>
python scripts/search_chunks.py "your query" --source-id <source-id>
```

Qdrant must be running (`docker compose up -d qdrant`).
