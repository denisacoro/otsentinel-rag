"""Step 14: full system evaluation across naive LLM, baseline dense RAG,
advanced hybrid RAG, and fine-tuned hybrid RAG, on the full golden evaluation
set (32 questions, data/eval/golden_eval_v1.jsonl).

RAGAS is deliberately skipped. Its faithfulness/relevance/correctness metrics
need an LLM judge; this project has no paid API budget, and a 3B model judging
its own family's outputs would be a weak, biased substitute not worth
reporting as a real number. In its place: deterministic citation-vs-reference
metrics (does the model cite the chunk_ids the golden set says are relevant),
refusal-vs-answerable accuracy, latency, and a stratified sample file for
manual read-through of a subset of answers -- this project's established
"no invented numbers" standard applied to evaluation itself.

Retrieval-quality metrics (Recall@K, MRR, nDCG for dense vs hybrid vs
hybrid+rerank) were already measured in step 10 -- see
docs/RETRIEVAL_SELECTION.md. This script does not recompute those. It measures
what step 10 did not: end-to-end ANSWER quality across different generators on
the same frozen test set.

Systems evaluated:
  A. naive            -- base model, no retrieval, plain question-answering
                          prompt. No citations expected, not PASS/FAIL scored --
                          included only as a qualitative hallucination-risk
                          contrast (this is what RAG replaces).
  B. baseline_dense    -- dense-only retrieval + base model.
  C. advanced_hybrid   -- hybrid retrieval (current production default) + base
                          model.
  D. finetuned_hybrid  -- hybrid retrieval (same retrieval as C) + QLoRA
                          adapter. Included for the complete, documented
                          comparison the roadmap asks for; the adapter itself
                          is NOT deployed (see docs/FINETUNING_RESULTS.md).

No source_id filtering is applied to retrieval for any system -- golden_eval
questions carry relevant_document_ids (plural, sometimes 2 for multi-document
questions), not a single filterable source_id, and filtering to "the right
document" would test something other than real retrieval.

Note on the validation/test split: retrieval CONFIGURATION (hybrid, no
reranker) was chosen in step 10 using the validation split. All four systems
here use that same fixed retrieval configuration, so that earlier choice does
not advantage one system over another in this comparison. Both splits (23
validation + 9 test) are used together for statistical power on a 32-question
set; this is noted explicitly in the report rather than silently overlooked.

Requires Qdrant running (docker compose up -d qdrant) and Ollama running with
the configured generator model.

Usage:
    python scripts/run_full_evaluation.py
"""

from __future__ import annotations

# unsloth must be imported before trl/transformers/peft are touched anywhere
# in this process -- see https://github.com/unslothai/unsloth/issues/2797.
# This script always ends up loading the adapter (system D), so unsloth is
# imported unconditionally, first, before any app.retrieval import (which
# pulls in transformers via sentence-transformers).
from unsloth import FastLanguageModel  # noqa: F401  (import-order side effect only)

import gc
import json
import re
import time
from pathlib import Path
from statistics import mean

import torch

from app.core.settings import get_settings
from app.generation.adapter_client import AdapterClient
from app.generation.llm_client import OllamaClient
from app.generation.prompts import REFUSAL_TEXT, build_messages
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse_embedder import SparseEmbedder
from app.retrieval.vector_store import create_qdrant_client, search_dense_chunks, search_hybrid_chunks
from app.schemas.document import DocumentChunk
from app.schemas.eval import EvalQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "golden_eval_v1.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "full_evaluation"
RESULTS_PATH = OUTPUT_DIR / "results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
MANUAL_REVIEW_PATH = OUTPUT_DIR / "manual_review_sample.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / "phase2_checkpoint.json"
# Phase 3 (loading the adapter via Unsloth) is the only phase that has failed
# in practice -- on Windows, memory-mapped safetensors loading can hit
# "OSError: The paging file is too small" under RAM pressure, unrelated to
# anything in this script. Phase 1+2 (retrieval + 96 Ollama generations) take
# far longer to redo than phase 3, so their results are checkpointed to disk
# before phase 3 starts. If this checkpoint exists, phase 1+2 are skipped
# entirely on the next run and phase 3 is retried directly. Delete
# CHECKPOINT_PATH manually to force a full rerun from scratch.

TOP_K = 5
CONFIG_NAME = "section-bge-m3-512-64"
ADAPTER_PATH = PROJECT_ROOT / "models" / "qlora_v1_adapter"

NAIVE_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following question about "
    "OT/SCADA/ICS/industrial security to the best of your knowledge, in the "
    "same language as the question. Do not mention that you lack access to "
    "documents or a knowledge base; just answer directly and concisely."
)

CITATION_TAG_PATTERN = re.compile(r"\[S(\d+)\]")

SYSTEMS = ["naive", "baseline_dense", "advanced_hybrid", "finetuned_hybrid"]


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from rag_pipeline.py rather than imported, since
# importing rag_pipeline at module load time would pull in transformers
# before unsloth gets a chance to patch it -- same reasoning as
# compare_qlora_finetuning.py).
# ---------------------------------------------------------------------------


def payload_to_chunk(payload: dict) -> DocumentChunk:
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
    cited: list[str] = []
    for number in CITATION_TAG_PATTERN.findall(answer_text):
        tag = f"S{number}"
        chunk_id = tag_to_chunk_id.get(tag)
        if chunk_id and chunk_id not in cited:
            cited.append(chunk_id)
    return cited


def load_golden_eval(path: Path) -> list[EvalQuestion]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(EvalQuestion.model_validate(json.loads(line)))
    return questions


# ---------------------------------------------------------------------------
# Phase 1: retrieval (no LLM loaded)
# ---------------------------------------------------------------------------


def prepare_retrieval(questions: list[EvalQuestion]) -> dict[str, dict]:
    """Runs dense and hybrid retrieval for every question up front, before any
    generator is loaded. Returns {question_id: {"dense": entry, "hybrid": entry}}
    where each entry has messages/tag_to_chunk_id/retrieval_refused/top_score.
    """
    settings = get_settings()

    dense_embedder = DenseEmbedder(
        model_name=settings.embedding_model_name,
        requested_device=settings.embedding_device,
    )
    sparse_embedder = SparseEmbedder()
    client = create_qdrant_client(settings.qdrant_url)

    prepared: dict[str, dict] = {}
    for q in questions:
        dense_vector = dense_embedder.encode_query(q.question)
        sparse_vector = sparse_embedder.encode_query(q.question)

        dense_results = search_dense_chunks(
            client=client,
            collection_name=settings.qdrant_collection,
            query_vector=dense_vector,
            top_k=TOP_K,
            config_name=CONFIG_NAME,
        )
        hybrid_results = search_hybrid_chunks(
            client=client,
            collection_name=settings.qdrant_collection,
            dense_query_vector=dense_vector,
            sparse_query_vector=sparse_vector,
            top_k=TOP_K,
            prefetch_limit=settings.hybrid_prefetch_limit,
            config_name=CONFIG_NAME,
        )

        prepared[q.question_id] = {
            "dense": _build_retrieval_entry(dense_results, q, settings),
            "hybrid": _build_retrieval_entry(hybrid_results, q, settings),
        }

    del dense_embedder, sparse_embedder
    gc.collect()
    torch.cuda.empty_cache()

    return prepared


def _build_retrieval_entry(results, q: EvalQuestion, settings) -> dict:
    top_score = results[0].score if results else 0.0
    retrieved_chunk_ids = [r.payload["chunk_id"] for r in results if r.payload]

    if not results or top_score < settings.min_retrieval_score:
        return {
            "retrieval_refused": True,
            "messages": None,
            "tag_to_chunk_id": {},
            "top_score": top_score,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }

    chunks = [payload_to_chunk(r.payload or {}) for r in results]
    messages, tag_to_chunk_id = build_messages(question=q.question, language=q.language, chunks=chunks)
    return {
        "retrieval_refused": False,
        "messages": messages,
        "tag_to_chunk_id": tag_to_chunk_id,
        "top_score": top_score,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


# ---------------------------------------------------------------------------
# Phase 2/3: generation
# ---------------------------------------------------------------------------


def generate_rag_answer(generator, retrieval_entry: dict) -> dict:
    if retrieval_entry["retrieval_refused"]:
        return {
            "answer_text": REFUSAL_TEXT,
            "refused": True,
            "cited_chunk_ids": [],
            "generation_latency_ms": 0.0,
        }

    answer_text, latency_ms = generator.chat(retrieval_entry["messages"])
    cited_chunk_ids = extract_cited_chunk_ids(answer_text, retrieval_entry["tag_to_chunk_id"])
    return {
        "answer_text": answer_text,
        "refused": REFUSAL_TEXT in answer_text,
        "cited_chunk_ids": cited_chunk_ids,
        "generation_latency_ms": latency_ms,
    }


def generate_naive_answer(generator, question: EvalQuestion) -> dict:
    messages = [
        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
        {"role": "user", "content": question.question},
    ]
    answer_text, latency_ms = generator.chat(messages)
    return {
        "answer_text": answer_text,
        "refused": REFUSAL_TEXT in answer_text,
        "cited_chunk_ids": [],
        "generation_latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_citations(cited_chunk_ids: list[str], relevant_chunk_ids: list[str]) -> dict:
    if not relevant_chunk_ids:
        return {"citation_precision": None, "citation_recall": None}
    cited = set(cited_chunk_ids)
    relevant = set(relevant_chunk_ids)
    overlap = cited & relevant
    precision = (len(overlap) / len(cited)) if cited else 0.0
    recall = len(overlap) / len(relevant)
    return {"citation_precision": precision, "citation_recall": recall}


def score_refusal(*, refused: bool, answerable: bool) -> str:
    if refused and answerable:
        return "FAIL_over_refusal"
    if not refused and not answerable:
        return "FAIL_answered_unanswerable"
    return "PASS"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(int(len(s) * pct), len(s) - 1)
    return s[idx]


def summarize_system(system_name: str, rows: list[dict]) -> dict:
    gen_latencies = [r["generation_latency_ms"] for r in rows if r["generation_latency_ms"] > 0]
    total_latencies = [r["retrieval_latency_ms"] + r["generation_latency_ms"] for r in rows]
    precisions = [r["citation_precision"] for r in rows if r.get("citation_precision") is not None]
    recalls = [r["citation_recall"] for r in rows if r.get("citation_recall") is not None]
    verdicts = [r["verdict"] for r in rows if r.get("verdict")]

    summary = {
        "system": system_name,
        "n_questions": len(rows),
        "avg_citation_precision": round(mean(precisions), 3) if precisions else None,
        "avg_citation_recall": round(mean(recalls), 3) if recalls else None,
        "pass_rate": round(verdicts.count("PASS") / len(verdicts), 3) if verdicts else None,
        "fail_over_refusal": verdicts.count("FAIL_over_refusal"),
        "fail_answered_unanswerable": verdicts.count("FAIL_answered_unanswerable"),
        "refusal_rate": round(sum(1 for r in rows if r["refused"]) / len(rows), 3) if rows else None,
        "generation_latency_p50_ms": percentile(gen_latencies, 0.5),
        "generation_latency_p95_ms": percentile(gen_latencies, 0.95),
        "total_latency_p50_ms": percentile(total_latencies, 0.5),
        "total_latency_p95_ms": percentile(total_latencies, 0.95),
    }
    return summary


def build_manual_review_sample(all_rows: dict[str, list[dict]], questions: list[EvalQuestion]) -> list[dict]:
    """Every 4th question by list order (8 of 32), all 4 systems' answers
    side by side, so a human reviewer can compare systems on the same
    question rather than scattered independent samples."""
    sampled_questions = questions[::4]
    sample = []
    for q in sampled_questions:
        entry = {
            "question_id": q.question_id,
            "question": q.question,
            "language": q.language,
            "question_type": q.question_type,
            "answerable": q.answerable,
            "reference_answer": q.reference_answer,
            "answers": {},
        }
        for system in SYSTEMS:
            row = next(r for r in all_rows[system] if r["question_id"] == q.question_id)
            entry["answers"][system] = {
                "answer_text": row["answer_text"],
                "cited_chunk_ids": row.get("cited_chunk_ids", []),
                "verdict": row.get("verdict"),
            }
        sample.append(entry)
    return sample


# ---------------------------------------------------------------------------
# Checkpointing (phase 1+2 results, so a phase-3 crash never loses them)
# ---------------------------------------------------------------------------


def save_checkpoint(all_rows: dict[str, list[dict]], retrieval: dict[str, dict], questions: list[EvalQuestion]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # only the hybrid retrieval entries are needed for phase 3
    hybrid_only = {qid: entry["hybrid"] for qid, entry in retrieval.items()}
    payload = {
        "all_rows": {system: all_rows[system] for system in ("naive", "baseline_dense", "advanced_hybrid")},
        "hybrid_retrieval": hybrid_only,
        "question_ids": [q.question_id for q in questions],
    }
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Checkpoint saved to {CHECKPOINT_PATH} (phase 1+2 results, safe to resume from if phase 3 fails)")


def load_checkpoint() -> tuple[dict[str, list[dict]], dict[str, dict]] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    all_rows: dict[str, list[dict]] = {system: [] for system in SYSTEMS}
    all_rows.update(payload["all_rows"])
    retrieval = {qid: {"hybrid": entry} for qid, entry in payload["hybrid_retrieval"].items()}
    print(f"Resuming from checkpoint {CHECKPOINT_PATH} -- skipping phase 1+2 (already completed).")
    return all_rows, retrieval


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    questions = load_golden_eval(GOLDEN_EVAL_PATH)
    print(f"Loaded {len(questions)} golden evaluation questions")

    checkpoint = load_checkpoint()
    if checkpoint is not None:
        all_rows, retrieval = checkpoint
    else:
        print("\nPhase 1: running dense + hybrid retrieval for all questions...")
        retrieval = prepare_retrieval(questions)
        hybrid_refused = sum(1 for e in retrieval.values() if e["hybrid"]["retrieval_refused"])
        dense_refused = sum(1 for e in retrieval.values() if e["dense"]["retrieval_refused"])
        print(f"Retrieval-layer refusals -- dense: {dense_refused}/{len(questions)}  hybrid: {hybrid_refused}/{len(questions)}")

        all_rows = {system: [] for system in SYSTEMS}

        print("\nPhase 2: base model via Ollama -- naive, baseline_dense, advanced_hybrid...")
        settings = get_settings()
        base_generator = OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.generator_model_name,
            temperature=settings.generation_temperature,
        )

        for q in questions:
            naive_result = generate_naive_answer(base_generator, q)
            all_rows["naive"].append(
                {
                    "question_id": q.question_id,
                    "answer_text": naive_result["answer_text"],
                    "refused": naive_result["refused"],
                    "cited_chunk_ids": [],
                    "citation_precision": None,
                    "citation_recall": None,
                    "verdict": None,  # not scored -- see module docstring
                    "retrieval_latency_ms": 0.0,
                    "generation_latency_ms": naive_result["generation_latency_ms"],
                }
            )

            for system, mode in (("baseline_dense", "dense"), ("advanced_hybrid", "hybrid")):
                entry = retrieval[q.question_id][mode]
                result = generate_rag_answer(base_generator, entry)
                citation_scores = score_citations(result["cited_chunk_ids"], q.relevant_chunk_ids)
                verdict = score_refusal(refused=result["refused"], answerable=q.answerable)
                all_rows[system].append(
                    {
                        "question_id": q.question_id,
                        "answer_text": result["answer_text"],
                        "refused": result["refused"],
                        "cited_chunk_ids": result["cited_chunk_ids"],
                        **citation_scores,
                        "verdict": verdict,
                        "retrieval_latency_ms": 0.0,  # measured once in step 10; not re-timed here
                        "generation_latency_ms": result["generation_latency_ms"],
                    }
                )
            print(f"  {q.question_id} done (naive, dense, hybrid)")

        save_checkpoint(all_rows, retrieval, questions)

    print("\nPhase 3: fine-tuned adapter via Unsloth -- finetuned_hybrid...")
    try:
        # max_seq_length=3584: the adapter's own default (2560) was tuned to the
        # ~2400-token prompts seen in the 12-question smoke set
        # (compare_qlora_finetuning.py); this 32-question golden set has real
        # top_k=5 retrieval prompts reaching 2689 tokens, which silently
        # truncated 2/32 questions on the first run here. 3584 covers the
        # longest observed prompt (2689) plus max_new_tokens=400 of generation
        # with margin for other questions' chunk-length variability.
        adapter_generator = AdapterClient(model_path=ADAPTER_PATH, max_seq_length=3584)

        for q in questions:
            entry = retrieval[q.question_id]["hybrid"]
            result = generate_rag_answer(adapter_generator, entry)
            citation_scores = score_citations(result["cited_chunk_ids"], q.relevant_chunk_ids)
            verdict = score_refusal(refused=result["refused"], answerable=q.answerable)
            all_rows["finetuned_hybrid"].append(
                {
                    "question_id": q.question_id,
                    "answer_text": result["answer_text"],
                    "refused": result["refused"],
                    "cited_chunk_ids": result["cited_chunk_ids"],
                    **citation_scores,
                    "verdict": verdict,
                    "retrieval_latency_ms": 0.0,
                    "generation_latency_ms": result["generation_latency_ms"],
                }
            )
            print(f"  {q.question_id} done (finetuned)")

        adapter_generator.unload()
    except Exception as exc:
        print(f"\nPhase 3 failed: {exc!r}")
        print(
            "Phase 1+2 results are safely checkpointed -- rerun this script to retry "
            "phase 3 only (it will skip straight past phase 1+2). Saving phase 1+2 "
            "results now so they're on disk regardless."
        )
        if not all_rows["finetuned_hybrid"]:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with RESULTS_PATH.open("w", encoding="utf-8") as f:
                for system in ("naive", "baseline_dense", "advanced_hybrid"):
                    for row in all_rows[system]:
                        f.write(json.dumps({"system": system, **row}, ensure_ascii=False) + "\n")
            print(f"Partial results (phase 1+2 only) saved to {RESULTS_PATH}")
        raise

    # ---- aggregate, save, print ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for system in SYSTEMS:
            for row in all_rows[system]:
                f.write(json.dumps({"system": system, **row}, ensure_ascii=False) + "\n")
    print(f"\nFull per-question results saved to {RESULTS_PATH}")

    summaries = [summarize_system(system, all_rows[system]) for system in SYSTEMS]
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to {SUMMARY_PATH}")

    manual_sample = build_manual_review_sample(all_rows, questions)
    with MANUAL_REVIEW_PATH.open("w", encoding="utf-8") as f:
        for entry in manual_sample:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Manual review sample ({len(manual_sample)} questions x {len(SYSTEMS)} systems) saved to {MANUAL_REVIEW_PATH}")

    def fmt(x, width):
        # round before stringifying -- raw float reprs (e.g. 6813.02979999998)
        # are long enough to blow past fixed column widths and run into the
        # next column with no separator.
        text = "None" if x is None else str(round(x, 1))
        return text.ljust(width)

    print(f"\n{'system':<20}{'pass_rate':<12}{'cit_prec':<10}{'cit_rec':<10}{'refusal%':<10}{'gen_p50ms':<12}{'gen_p95ms':<12}")
    for s in summaries:
        row = (
            s["system"].ljust(20)
            + fmt(s["pass_rate"], 12)
            + fmt(s["avg_citation_precision"], 10)
            + fmt(s["avg_citation_recall"], 10)
            + fmt(s["refusal_rate"], 10)
            + fmt(s["generation_latency_p50_ms"], 12)
            + fmt(s["generation_latency_p95_ms"], 12)
        )
        print(row)


if __name__ == "__main__":
    main()
