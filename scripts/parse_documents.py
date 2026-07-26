from __future__ import annotations

import argparse
from pathlib import Path

from app.ingestion.downloader import load_manifest
from app.ingestion.pdf_parser import (
    parse_pdf,
    save_pages_jsonl,
    save_parse_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sources.jsonl"

DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"

DEFAULT_PAGES_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "pages"

DEFAULT_REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "parse_reports"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Parse downloaded OTSentinel AI PDF documents.")

    parser.add_argument(
        "--source-id",
        help="Parse only the source with this source ID.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing parsed output.",
    )

    return parser.parse_args()


def main() -> None:
    """Parse enabled PDF sources from the source manifest."""

    arguments = parse_arguments()
    sources = load_manifest(DEFAULT_MANIFEST_PATH)

    completed_count = 0
    skipped_count = 0

    for source in sources:
        if not source.enabled:
            continue

        if arguments.source_id is not None and source.source_id != arguments.source_id:
            continue

        if Path(source.filename).suffix.lower() != ".pdf":
            print(f"Skipping non-PDF source: {source.source_id}")
            skipped_count += 1
            continue

        publisher_directory = source.publisher.lower()

        input_path = DEFAULT_RAW_DIRECTORY / publisher_directory / source.filename

        pages_output_path = DEFAULT_PAGES_DIRECTORY / f"{source.source_id}.pages.jsonl"

        report_output_path = DEFAULT_REPORTS_DIRECTORY / f"{source.source_id}.summary.json"

        if pages_output_path.exists() and report_output_path.exists() and not arguments.force:
            print(f"Already parsed: {source.source_id}\nUse --force to parse it again.")
            skipped_count += 1
            continue

        print(f"Parsing: {source.source_id}")
        print(f"  Input: {input_path}")

        pages, summary = parse_pdf(
            source_id=source.source_id,
            document_id=source.source_id,
            title=source.title,
            pdf_path=input_path,
        )

        save_pages_jsonl(
            pages=pages,
            output_path=pages_output_path,
        )

        save_parse_summary(
            summary=summary,
            output_path=report_output_path,
        )

        print(f"  Total pages: {summary.total_pages}")
        print(f"  Pages with text: {summary.pages_with_text}")
        print(f"  Empty pages: {summary.empty_pages}")
        print(f"  Total words: {summary.total_words}")
        print(f"  Page records: {pages_output_path}")
        print(f"  Report: {report_output_path}")

        completed_count += 1

    print(f"\nParsing completed.\n  Parsed: {completed_count}\n  Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
