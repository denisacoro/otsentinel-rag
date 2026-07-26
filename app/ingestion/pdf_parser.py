from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from app.ingestion.downloader import calculate_sha256
from app.ingestion.text_cleaner import normalize_extracted_text
from app.schemas.document import ParsedPage, PdfParseSummary


def count_words(text: str) -> int:
    """Count whitespace-separated words in normalized text."""

    return len(text.split())


def parse_pdf(
    *,
    source_id: str,
    document_id: str,
    title: str,
    pdf_path: Path,
) -> tuple[list[ParsedPage], PdfParseSummary]:
    """Extract and normalize text from every page of a PDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    parsed_pages: list[ParsedPage] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            raw_text = page.get_text(
                "text",
                sort=True,
            )

            normalized_text = normalize_extracted_text(raw_text)

            parsed_page = ParsedPage(
                source_id=source_id,
                document_id=document_id,
                title=title,
                page_number=page_index + 1,
                text=normalized_text,
                character_count=len(normalized_text),
                word_count=count_words(normalized_text),
                is_empty=not bool(normalized_text),
            )

            parsed_pages.append(parsed_page)

    pages_with_text = sum(not parsed_page.is_empty for parsed_page in parsed_pages)
    empty_pages = sum(parsed_page.is_empty for parsed_page in parsed_pages)

    summary = PdfParseSummary(
        source_id=source_id,
        document_id=document_id,
        title=title,
        input_path=str(pdf_path),
        file_sha256=calculate_sha256(pdf_path),
        file_size_bytes=pdf_path.stat().st_size,
        total_pages=len(parsed_pages),
        pages_with_text=pages_with_text,
        empty_pages=empty_pages,
        total_characters=sum(page.character_count for page in parsed_pages),
        total_words=sum(page.word_count for page in parsed_pages),
        parsed_at_utc=datetime.now(UTC),
    )

    return parsed_pages, summary


def save_pages_jsonl(
    pages: list[ParsedPage],
    output_path: Path,
) -> None:
    """Save parsed pages as one JSON object per line."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for page in pages:
                record = page.model_dump(mode="json")

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )
                output_file.write("\n")

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_parse_summary(
    summary: PdfParseSummary,
    output_path: Path,
) -> None:
    """Save a human-readable JSON parsing report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
