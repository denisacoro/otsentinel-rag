from __future__ import annotations

import argparse
import re
from pathlib import Path

from transformers import AutoTokenizer

from app.ingestion.chunker import (
    build_document_chunks,
    load_sections_jsonl,
    save_chunking_summary,
    save_chunks_jsonl,
)
from app.ingestion.downloader import load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sources.jsonl"

SECTIONS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "sections"

CHUNKS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "chunks"

REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "chunk_reports"


def sanitize_config_name(value: str) -> str:
    """Convert a configuration name into a safe file component."""

    sanitized = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        value,
    )

    return sanitized.strip("-").lower()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Create structure-aware, token-limited retrieval chunks.")
    )

    parser.add_argument(
        "--source-id",
        help="Process only this source ID.",
    )

    parser.add_argument(
        "--tokenizer-name",
        default="BAAI/bge-m3",
        help="Hugging Face tokenizer model ID.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens in embedding_text.",
    )

    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=64,
        help="Body-token overlap between adjacent chunks.",
    )

    parser.add_argument(
        "--config-name",
        default="section-bge-m3-512-64",
        help="Name used to identify this chunking run.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing chunk output.",
    )

    return parser.parse_args()


def main() -> None:
    """Create chunks for enabled manifest sources."""

    arguments = parse_arguments()

    config_name = sanitize_config_name(arguments.config_name)

    print(f"Loading tokenizer: {arguments.tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.tokenizer_name,
        use_fast=True,
    )

    model_max_length = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    if (
        isinstance(model_max_length, int)
        and model_max_length > 0
        and model_max_length < arguments.max_tokens
    ):
        raise ValueError(
            f"Requested {arguments.max_tokens} tokens, but "
            f"the tokenizer declares a maximum of "
            f"{model_max_length}."
        )

    sources = load_manifest(MANIFEST_PATH)

    completed_count = 0
    skipped_count = 0

    for source in sources:
        if not source.enabled:
            continue

        if arguments.source_id is not None and source.source_id != arguments.source_id:
            continue

        sections_path = SECTIONS_DIRECTORY / f"{source.source_id}.sections.jsonl"

        chunks_path = CHUNKS_DIRECTORY / (f"{source.source_id}.{config_name}.chunks.jsonl")

        report_path = REPORTS_DIRECTORY / (f"{source.source_id}.{config_name}.summary.json")

        if chunks_path.exists() and report_path.exists() and not arguments.force:
            print(f"Chunks already exist: {source.source_id}\nUse --force to regenerate them.")

            skipped_count += 1
            continue

        print(f"Chunking: {source.source_id}")
        print(f"  Sections: {sections_path}")

        sections = load_sections_jsonl(sections_path)

        chunks, summary = build_document_chunks(
            source_id=source.source_id,
            document_id=source.source_id,
            title=source.title,
            sections=sections,
            sections_path=sections_path,
            tokenizer=tokenizer,
            tokenizer_name=arguments.tokenizer_name,
            config_name=config_name,
            max_tokens=arguments.max_tokens,
            overlap_tokens=arguments.overlap_tokens,
        )

        save_chunks_jsonl(
            chunks=chunks,
            output_path=chunks_path,
        )

        save_chunking_summary(
            summary=summary,
            output_path=report_path,
        )

        print(f"  Sections: {summary.total_sections}")
        print(f"  Sections with content: {summary.sections_with_content}")
        print(f"  Chunks: {summary.total_chunks}")
        print(f"  Token range: {summary.minimum_chunk_tokens}-{summary.maximum_chunk_tokens}")
        print(f"  Average tokens: {summary.average_chunk_tokens:.2f}")
        print(f"  Output: {chunks_path}")
        print(f"  Report: {report_path}")

        completed_count += 1

    print(f"\nChunking completed.\n  Processed: {completed_count}\n  Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
