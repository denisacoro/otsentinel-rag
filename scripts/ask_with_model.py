"""CLI: ask a question through the RAG pipeline, choosing which generator
answers it -- base model via Ollama (production default) or the QLoRA adapter
via Unsloth (evaluation only). This is step 13's "model selector," implemented
without vLLM.

Usage:
    python scripts/ask_with_model.py "What is a physical access control system?" --model base
    python scripts/ask_with_model.py "What is a physical access control system?" --model finetuned
    python scripts/ask_with_model.py "Ce este un sistem SCADA?" --model finetuned --language ro
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--model", choices=["base", "finetuned"], default="base")
    parser.add_argument("--language", default="en", choices=["en", "ro"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-id", default=None)
    args = parser.parse_args()

    generator = None
    if args.model == "finetuned":
        # unsloth must be imported before anything below touches transformers
        # (via app.retrieval.embedder.DenseEmbedder -> sentence-transformers)
        from app.generation.adapter_client import AdapterClient

        print("Loading fine-tuned adapter (models/qlora_v1_adapter)...")
        generator = AdapterClient()

    from app.generation.rag_pipeline import answer_question

    result = answer_question(
        question=args.question,
        language=args.language,
        top_k=args.top_k,
        source_id=args.source_id,
        generator=generator,
    )

    print(f"\nModel: {result.model_name}")
    print(f"Refused: {result.refused}")
    print(f"\nAnswer:\n{result.answer_text}")
    print(f"\nCited chunks: {result.cited_chunk_ids}")
    print(
        f"Retrieval latency: {result.retrieval_latency_ms:.0f}ms  "
        f"Generation latency: {result.generation_latency_ms:.0f}ms"
    )


if __name__ == "__main__":
    main()
