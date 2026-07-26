import json
from pathlib import Path

import pymupdf

from app.ingestion.pdf_parser import (
    parse_pdf,
    save_pages_jsonl,
)
from app.ingestion.text_cleaner import normalize_extracted_text


def create_test_pdf(pdf_path: Path) -> None:
    """Create a small two-page PDF for parser tests."""

    document = pymupdf.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "OT security",
    )
    first_page.insert_text(
        (72, 92),
        "Network segmentation",
    )

    document.new_page()

    document.save(pdf_path)
    document.close()


def test_normalize_extracted_text() -> None:
    raw_text = "First line  \r\n\r\n\r\n\r\nSecond\u00a0line\u00ad"

    normalized = normalize_extracted_text(raw_text)

    assert normalized == "First line\n\nSecond line"


def test_parse_pdf_extracts_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "example.pdf"
    create_test_pdf(pdf_path)

    pages, summary = parse_pdf(
        source_id="test-source",
        document_id="test-document",
        title="Test Document",
        pdf_path=pdf_path,
    )

    assert len(pages) == 2

    assert pages[0].page_number == 1
    assert "OT security" in pages[0].text
    assert "Network segmentation" in pages[0].text
    assert pages[0].is_empty is False

    assert pages[1].page_number == 2
    assert pages[1].is_empty is True

    assert summary.total_pages == 2
    assert summary.pages_with_text == 1
    assert summary.empty_pages == 1
    assert summary.file_size_bytes > 0
    assert len(summary.file_sha256) == 64


def test_save_pages_jsonl(tmp_path: Path) -> None:
    pdf_path = tmp_path / "example.pdf"
    output_path = tmp_path / "pages.jsonl"

    create_test_pdf(pdf_path)

    pages, _ = parse_pdf(
        source_id="test-source",
        document_id="test-document",
        title="Test Document",
        pdf_path=pdf_path,
    )

    save_pages_jsonl(
        pages=pages,
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    first_record = json.loads(lines[0])

    assert first_record["page_number"] == 1
    assert first_record["source_id"] == "test-source"
