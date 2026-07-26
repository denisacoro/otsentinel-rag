import json
from pathlib import Path

import pymupdf

from app.ingestion.structure_extractor import (
    extract_document_structure,
    save_models_jsonl,
)


def create_structured_test_pdf(pdf_path: Path) -> None:
    """Create a PDF with headings and repeated margins."""

    document = pymupdf.open()

    for page_index in range(5):
        page = document.new_page(
            width=595,
            height=842,
        )

        page.insert_text(
            (72, 30),
            "NIST TEST PUBLICATION",
            fontsize=8,
            fontname="helv",
        )

        page.insert_text(
            (72, 815),
            f"Page {page_index + 1}",
            fontsize=8,
            fontname="helv",
        )

        if page_index == 0:
            page.insert_text(
                (72, 100),
                "1 Introduction",
                fontsize=18,
                fontname="hebo",
            )

            page.insert_text(
                (72, 135),
                "This document explains OT security.",
                fontsize=11,
                fontname="helv",
            )

        elif page_index == 1:
            page.insert_text(
                (72, 100),
                "1.1 Purpose",
                fontsize=15,
                fontname="hebo",
            )

            page.insert_text(
                (72, 135),
                "The purpose is to protect industrial systems.",
                fontsize=11,
                fontname="helv",
            )

        else:
            page.insert_text(
                (72, 100),
                "Additional body content for the section.",
                fontsize=11,
                fontname="helv",
            )

    document.save(pdf_path)
    document.close()


def test_removes_repeated_headers_and_footers(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "structured.pdf"
    create_structured_test_pdf(pdf_path)

    sections, summary, lines = extract_document_structure(
        source_id="test-source",
        document_id="test-document",
        title="Test Document",
        pdf_path=pdf_path,
    )

    assert summary.repeated_margin_lines == 10

    repeated_texts = {line.text for line in lines if line.is_repeated_margin}

    assert "NIST TEST PUBLICATION" in repeated_texts
    assert "Page 1" in repeated_texts

    section_content = "\n".join(section.text for section in sections)

    assert "NIST TEST PUBLICATION" not in section_content
    assert "Page 1" not in section_content


def test_detects_heading_hierarchy(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "structured.pdf"
    create_structured_test_pdf(pdf_path)

    sections, summary, _ = extract_document_structure(
        source_id="test-source",
        document_id="test-document",
        title="Test Document",
        pdf_path=pdf_path,
    )

    assert summary.detected_headings >= 2

    sections_by_heading = {section.heading: section for section in sections}

    introduction = sections_by_heading["1 Introduction"]
    purpose = sections_by_heading["1.1 Purpose"]

    assert introduction.heading_level == 1
    assert introduction.heading_path == ["1 Introduction"]

    assert purpose.heading_level == 2
    assert purpose.heading_path == [
        "1 Introduction",
        "1.1 Purpose",
    ]


def test_saves_sections_as_jsonl(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "structured.pdf"
    output_path = tmp_path / "sections.jsonl"

    create_structured_test_pdf(pdf_path)

    sections, _, _ = extract_document_structure(
        source_id="test-source",
        document_id="test-document",
        title="Test Document",
        pdf_path=pdf_path,
    )

    save_models_jsonl(
        records=sections,
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == len(sections)

    first_record = json.loads(lines[0])

    assert first_record["source_id"] == "test-source"
    assert first_record["section_id"].startswith("test-document::section::")
