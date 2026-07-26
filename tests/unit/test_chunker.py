import json
from pathlib import Path

import pytest

from app.ingestion.chunker import (
    build_document_chunks,
    create_section_chunks,
    save_chunks_jsonl,
)
from app.schemas.document import StructuredSection


class SimpleWhitespaceTokenizer:
    """Small offline tokenizer used only in unit tests."""

    model_max_length = 10_000

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens

        token_ids: list[int] = []

        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token

            token_ids.append(self.token_to_id[token])

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens
        del clean_up_tokenization_spaces

        return " ".join(self.id_to_token[token_id] for token_id in token_ids)


def create_test_section(
    *,
    text: str,
) -> StructuredSection:
    return StructuredSection(
        source_id="test-source",
        document_id="test-document",
        section_id="test-document::section::0001",
        heading="1.1 Network Segmentation",
        heading_level=2,
        heading_path=[
            "1 OT Security",
            "1.1 Network Segmentation",
        ],
        page_start=10,
        page_end=12,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


def test_creates_multiple_overlapping_chunks() -> None:
    tokenizer = SimpleWhitespaceTokenizer()

    text = " ".join(f"token-{index}" for index in range(40))

    section = create_test_section(text=text)

    chunks = create_section_chunks(
        section=section,
        title="Test OT Document",
        tokenizer=tokenizer,
        tokenizer_name="test-tokenizer",
        config_name="test-24-3",
        max_tokens=24,
        overlap_tokens=3,
    )
    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.token_count <= 24
        assert chunk.heading == ("1.1 Network Segmentation")
        assert chunk.heading_path == [
            "1 OT Security",
            "1.1 Network Segmentation",
        ]
        assert chunk.page_start == 10
        assert chunk.page_end == 12

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert first_words[-3:] == second_words[:3]
    assert chunks[0].overlap_with_previous == 0
    assert chunks[1].overlap_with_previous == 3


def test_embedding_text_contains_section_context() -> None:
    tokenizer = SimpleWhitespaceTokenizer()

    section = create_test_section(
        text="OT networks should be segmented.",
    )

    chunks = create_section_chunks(
        section=section,
        title="Test OT Document",
        tokenizer=tokenizer,
        tokenizer_name="test-tokenizer",
        config_name="test-config",
        max_tokens=30,
        overlap_tokens=3,
    )

    assert len(chunks) == 1

    embedding_text = chunks[0].embedding_text

    assert "Document: Test OT Document" in embedding_text
    assert "1 OT Security" in embedding_text
    assert "1.1 Network Segmentation" in embedding_text
    assert "OT networks should be segmented." in embedding_text


def test_rejects_invalid_overlap() -> None:
    tokenizer = SimpleWhitespaceTokenizer()

    section = create_test_section(text="Example section content.")

    with pytest.raises(ValueError):
        create_section_chunks(
            section=section,
            title="Test Document",
            tokenizer=tokenizer,
            tokenizer_name="test-tokenizer",
            config_name="invalid",
            max_tokens=20,
            overlap_tokens=20,
        )


def test_saves_chunks_and_summary(
    tmp_path: Path,
) -> None:
    tokenizer = SimpleWhitespaceTokenizer()

    text = " ".join(f"word-{index}" for index in range(30))

    section = create_test_section(text=text)
    sections_path = tmp_path / "sections.jsonl"

    chunks, summary = build_document_chunks(
        source_id="test-source",
        document_id="test-document",
        title="Test OT Document",
        sections=[section],
        sections_path=sections_path,
        tokenizer=tokenizer,
        tokenizer_name="test-tokenizer",
        config_name="test-18-2",
        max_tokens=18,
        overlap_tokens=2,
    )

    output_path = tmp_path / "chunks.jsonl"

    save_chunks_jsonl(
        chunks=chunks,
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == len(chunks)

    first_record = json.loads(lines[0])

    assert first_record["source_id"] == "test-source"
    assert first_record["chunk_index"] == 1
    assert len(first_record["content_sha256"]) == 64

    assert summary.total_sections == 1
    assert summary.sections_with_content == 1
    assert summary.total_chunks == len(chunks)
    assert summary.maximum_chunk_tokens <= 18
