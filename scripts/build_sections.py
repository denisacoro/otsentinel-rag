from __future__ import annotations

import argparse
from pathlib import Path

from app.ingestion.downloader import load_manifest
from app.ingestion.structure_extractor import (
    extract_document_structure,
    save_models_jsonl,
    save_structure_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "sources.jsonl"

RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"

SECTIONS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "sections"

LAYOUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "layout_lines"

REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "structure_reports"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Remove repeated margins, detect headings and build structure-aware document sections."
        )
    )

    parser.add_argument(
        "--source-id",
        help="Process only this source ID.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing structure output.",
    )

    return parser.parse_args()


def main() -> None:
    """Build sections for enabled PDF sources."""

    arguments = parse_arguments()
    sources = load_manifest(MANIFEST_PATH)

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

        input_path = RAW_DIRECTORY / source.publisher.lower() / source.filename

        sections_path = SECTIONS_DIRECTORY / f"{source.source_id}.sections.jsonl"

        layout_path = LAYOUT_DIRECTORY / f"{source.source_id}.lines.jsonl"

        report_path = REPORTS_DIRECTORY / f"{source.source_id}.summary.json"

        if (
            sections_path.exists()
            and layout_path.exists()
            and report_path.exists()
            and not arguments.force
        ):
            print(f"Structure already exists: {source.source_id}\nUse --force to regenerate it.")

            skipped_count += 1
            continue

        print(f"Building structure: {source.source_id}")
        print(f"  Input: {input_path}")

        sections, summary, lines = extract_document_structure(
            source_id=source.source_id,
            document_id=source.source_id,
            title=source.title,
            pdf_path=input_path,
        )

        save_models_jsonl(
            records=sections,
            output_path=sections_path,
        )

        save_models_jsonl(
            records=lines,
            output_path=layout_path,
        )

        save_structure_summary(
            summary=summary,
            output_path=report_path,
        )

        print(f"  Total pages: {summary.total_pages}")
        print(f"  Total lines: {summary.total_lines}")
        print(f"  Repeated margin lines: {summary.repeated_margin_lines}")
        print(f"  Detected headings: {summary.detected_headings}")
        print(f"  Generated sections: {summary.generated_sections}")
        print(f"  Body font size: {summary.body_font_size}")
        print(f"  Sections: {sections_path}")
        print(f"  Layout audit: {layout_path}")
        print(f"  Report: {report_path}")

        completed_count += 1

    print(
        "\nStructure extraction completed.\n"
        f"  Processed: {completed_count}\n"
        f"  Skipped: {skipped_count}"
    )


if __name__ == "__main__":
    main()
