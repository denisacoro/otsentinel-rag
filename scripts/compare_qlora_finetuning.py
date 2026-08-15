"""Compare baseline vs QLoRA-fine-tuned generation on real retrieval results.

Runs the exact same hybrid retrieval the live pipeline uses (app.retrieval /
app.generation.prompts, unchanged) for each smoke-test question, then generates
an answer with (a) the unmodified base model and (b) the fine-tuned LoRA adapter,
using identical messages. This isolates what fine-tuning changed about generation
behavior (citation formatting, refusal correctness, Romanian consistency) from
retrieval quality, which fine-tuning does not touch.

Retrieval-layer refusal (score < min_retrieval_score) is replicated exactly as in
rag_pipeline.answer_question(): if retrieval itself refuses, neither model is
called, since both would see identical retrieval results anyway.

Models are loaded and generated with sequentially, not simultaneously -- 4GB VRAM
doesn't have room for two 3B models loaded at once.

Requires Qdrant running (docker compose up -d qdrant).

Usage:
    python scripts/compare_qlora_finetuning.py
"""

from __future__ import annotations

# unsloth must be imported before trl/transformers/peft -- see
# https://github.com/unslothai/unsloth/issues/2797
from unsloth import FastLanguageModel

import gc
import json
import re
from pathlib import Path

import torch

from app.core.settings import get_settings
from app.generation.prompts import REFUSAL_TEXT, build_messages
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse_embedder import SparseEmbedder
from app.retrieval.vector_store import create_qdrant_client, search_hybrid_chunks
from app.schemas.document import DocumentChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "smoke_questions.jsonl"
ADAPTER_PATH = PROJECT_ROOT / "models" / "qlora_v1_adapter"
BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "qlora_comparison" / "results.jsonl"
MAX_SEQ_LENGTH = 2560
TOP_K = 5
MAX_NEW_TOKENS = 300
CONFIG_NAME = "section-bge-m3-512-64"

CITATION_TAG_PATTERN = re.compile(r"\[S(\d+)\]")


def payload_to_chunk(payload: dict) -> DocumentChunk:
    """Rebuild a DocumentChunk from a Qdrant payload -- mirrors rag_pipeline._payload_to_chunk."""
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


def extract_cited_chunk_ids(answer_text: str, tag_to_chunk_id: dict[str, str]) -> list[str]:
    """Mirrors rag_pipeline.extract_cited_chunk_ids."""
    cited: list[str] = []
    for number in CITATION_TAG_PATTERN.findall(answer_text):
        tag = f"S{number}"
        chunk_id = tag_to_chunk_id.get(tag)
        if chunk_id and chunk_id not in cited:
            cited.append(chunk_id)
    return cited


def load_smoke_questions(path: Path) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def prepare_questions_with_retrieval(questions: list[dict]) -> list[dict]:
    """Phase 1: run real retrieval for every question before any LLM is loaded."""
    settings = get_settings()

    dense_embedder = DenseEmbedder(
        model_name=settings.embedding_model_name,
        requested_device=settings.embedding_device,
    )
    sparse_embedder = SparseEmbedder()
    client = create_qdrant_client(settings.qdrant_url)

    prepared: list[dict] = []
    for q in questions:
        dense_vector = dense_embedder.encode_query(q["question"])
        sparse_vector = sparse_embedder.encode_query(q["question"])

        results = search_hybrid_chunks(
            client=client,
            collection_name=settings.qdrant_collection,
            dense_query_vector=dense_vector,
            sparse_query_vector=sparse_vector,
            top_k=TOP_K,
            prefetch_limit=settings.hybrid_prefetch_limit,
            config_name=CONFIG_NAME,
            source_id=q.get("source_id"),
        )

        entry = {
            "question_id": q["question_id"],
            "question": q["question"],
            "language": q["language"],
            "answerable": q["answerable"],
            "top_score": results[0].score if results else 0.0,
        }

        if not results or results[0].score < settings.min_retrieval_score:
            entry["retrieval_refused"] = True
            entry["messages"] = None
            entry["tag_to_chunk_id"] = {}
        else:
            entry["retrieval_refused"] = False
            chunks = [payload_to_chunk(r.payload or {}) for r in results]
            messages, tag_to_chunk_id = build_messages(
                question=q["question"], language=q["language"], chunks=chunks
            )
            entry["messages"] = messages
            entry["tag_to_chunk_id"] = tag_to_chunk_id

        prepared.append(entry)

    # free the embedders before any LLM gets loaded
    del dense_embedder, sparse_embedder
    gc.collect()
    torch.cuda.empty_cache()

    return prepared


def generate_for_all(model_name: str, prepared: list[dict], label: str) -> dict[str, str]:
    """Phase 2/3: load one model, generate an answer for every question that needs one."""
    print(f"\nLoading {label} ({model_name})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    answers: dict[str, str] = {}
    for entry in prepared:
        if entry["retrieval_refused"]:
            answers[entry["question_id"]] = REFUSAL_TEXT
            continue

        input_ids = tokenizer.apply_chat_template(
            entry["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]
        answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        answers[entry["question_id"]] = answer_text
        print(f"  {entry['question_id']}: {len(answer_text)} chars")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return answers


def score_answer(*, answer_text: str, answerable: bool, tag_to_chunk_id: dict[str, str]) -> str:
    """PASS / FAIL / REVIEW, matching run_smoke_questions.py's criteria."""
    refused = REFUSAL_TEXT in answer_text
    cited = bool(extract_cited_chunk_ids(answer_text, tag_to_chunk_id))

    if refused and answerable:
        return "FAIL"
    if not refused and not answerable:
        return "FAIL"
    if not refused and not cited:
        return "REVIEW"
    return "PASS"


def main() -> None:
    questions = load_smoke_questions(SMOKE_QUESTIONS_PATH)
    print(f"Loaded {len(questions)} smoke questions")

    print("\nPhase 1: running real retrieval for all questions...")
    prepared = prepare_questions_with_retrieval(questions)
    refused_count = sum(1 for p in prepared if p["retrieval_refused"])
    print(f"Retrieval-layer refusals: {refused_count}/{len(prepared)}")

    baseline_answers = generate_for_all(BASE_MODEL_NAME, prepared, "BASELINE (no adapter)")
    finetuned_answers = generate_for_all(str(ADAPTER_PATH), prepared, "FINE-TUNED (qlora_v1_adapter)")

    results = []
    baseline_counts = {"PASS": 0, "FAIL": 0, "REVIEW": 0}
    finetuned_counts = {"PASS": 0, "FAIL": 0, "REVIEW": 0}

    print(f"\n{'question_id':<14}{'baseline':<10}{'finetuned':<10}")
    for entry in prepared:
        qid = entry["question_id"]
        tag_map = entry["tag_to_chunk_id"]

        baseline_verdict = score_answer(
            answer_text=baseline_answers[qid], answerable=entry["answerable"], tag_to_chunk_id=tag_map
        )
        finetuned_verdict = score_answer(
            answer_text=finetuned_answers[qid], answerable=entry["answerable"], tag_to_chunk_id=tag_map
        )
        baseline_counts[baseline_verdict] += 1
        finetuned_counts[finetuned_verdict] += 1

        print(f"{qid:<14}{baseline_verdict:<10}{finetuned_verdict:<10}")

        results.append(
            {
                "question_id": qid,
                "question": entry["question"],
                "language": entry["language"],
                "answerable": entry["answerable"],
                "retrieval_refused": entry["retrieval_refused"],
                "baseline_answer": baseline_answers[qid],
                "baseline_verdict": baseline_verdict,
                "finetuned_answer": finetuned_answers[qid],
                "finetuned_verdict": finetuned_verdict,
            }
        )

    print(f"\nBaseline:   {baseline_counts}")
    print(f"Fine-tuned: {finetuned_counts}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nFull results (both models' full answer text) saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()