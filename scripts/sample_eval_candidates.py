from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingestion.downloader import load_manifest
from app.retrieval.chunk_loader import load_chunks_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sources.jsonl"
CHUNKS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "chunks"
CANDIDATES_DIRECTORY = PROJECT_ROOT / "data" / "eval" / "candidates"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample representative chunks per source as material for golden eval questions."
    )
    parser.add_argument("--config-name", default="section-bge-m3-512-64")
    parser.add_argument("--sample-size", type=int, default=30, help="Chunks to sample per source.")
    return parser.parse_args()


def sample_evenly(chunks: list, sample_size: int) -> list:
    """Pick roughly evenly spaced chunks across the document for topical breadth."""

    if len(chunks) <= sample_size:
        return chunks

    step = len(chunks) / sample_size
    indices = [int(i * step) for i in range(sample_size)]

    return [chunks[i] for i in indices]


def main() -> None:
    arguments = parse_arguments()
    sources = load_manifest(MANIFEST_PATH)

    CANDIDATES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for source in sources:
        if not source.enabled:
            continue

        chunks_path = CHUNKS_DIRECTORY / f"{source.source_id}.{arguments.config_name}.chunks.jsonl"

        if not chunks_path.exists():
            print(f"Skipping {source.source_id}: no chunks found at {chunks_path}")
            continue

        chunks = load_chunks_jsonl(chunks_path)
        sampled = sample_evenly(chunks, arguments.sample_size)

        output_path = CANDIDATES_DIRECTORY / f"{source.source_id}.candidates.jsonl"

        with output_path.open("w", encoding="utf-8") as output_file:
            for chunk in sampled:
                output_file.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "title": chunk.title,
                            "heading": chunk.heading,
                            "heading_path": chunk.heading_path,
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "text": chunk.text,
                        }
                    )
                    + "\n"
                )

        print(f"{source.source_id}: sampled {len(sampled)}/{len(chunks)} chunks -> {output_path}")


if __name__ == "__main__":
    main()