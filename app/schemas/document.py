from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    """Text and metadata extracted from one PDF page."""

    source_id: str
    document_id: str
    title: str

    page_number: int = Field(ge=1)

    text: str
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    is_empty: bool

    extraction_method: Literal["pymupdf_text_sorted"] = "pymupdf_text_sorted"


class PdfParseSummary(BaseModel):
    """Summary produced after parsing one PDF document."""

    source_id: str
    document_id: str
    title: str

    input_path: str
    file_sha256: str
    file_size_bytes: int = Field(ge=0)

    total_pages: int = Field(ge=0)
    pages_with_text: int = Field(ge=0)
    empty_pages: int = Field(ge=0)

    total_characters: int = Field(ge=0)
    total_words: int = Field(ge=0)

    parsed_at_utc: datetime


class ExtractedLine(BaseModel):
    """One layout-aware line extracted from a PDF page."""

    source_id: str
    document_id: str
    page_number: int = Field(ge=1)

    text: str
    bbox: tuple[float, float, float, float]

    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)

    font_size: float = Field(gt=0)
    font_names: list[str]
    is_bold: bool

    position: Literal["header", "body", "footer"]

    is_repeated_margin: bool = False
    is_heading: bool = False
    heading_level: int | None = Field(default=None, ge=1, le=6)


class StructuredSection(BaseModel):
    """A logical document section created from detected headings."""

    source_id: str
    document_id: str
    section_id: str

    heading: str | None
    heading_level: int = Field(ge=0, le=6)
    heading_path: list[str]

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    text: str
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)


class StructureSummary(BaseModel):
    """Summary of layout analysis and section generation."""

    source_id: str
    document_id: str
    title: str

    input_path: str
    total_pages: int = Field(ge=0)
    total_lines: int = Field(ge=0)

    repeated_margin_lines: int = Field(ge=0)
    detected_headings: int = Field(ge=0)
    generated_sections: int = Field(ge=0)

    body_font_size: float = Field(gt=0)

    repeated_header_patterns: list[str]
    repeated_footer_patterns: list[str]

    processed_at_utc: datetime


class DocumentChunk(BaseModel):
    """One retrieval-ready chunk created from a structured section."""

    source_id: str
    document_id: str
    section_id: str
    chunk_id: str
    config_name: str

    title: str

    heading: str | None
    heading_level: int = Field(ge=0, le=6)
    heading_path: list[str]

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    chunk_index: int = Field(ge=1)
    chunk_count: int = Field(ge=1)

    text: str
    embedding_text: str

    body_token_count: int = Field(ge=1)
    token_count: int = Field(ge=1)

    token_start: int = Field(ge=0)
    token_end: int = Field(ge=1)

    overlap_with_previous: int = Field(ge=0)

    tokenizer_name: str
    strategy: Literal["section_token_window_v1"] = "section_token_window_v1"

    content_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


class ChunkingSummary(BaseModel):
    """Summary of one document chunking run."""

    source_id: str
    document_id: str
    title: str
    config_name: str

    input_sections_path: str
    tokenizer_name: str
    strategy: str

    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)

    total_sections: int = Field(ge=0)
    sections_with_content: int = Field(ge=0)
    empty_sections_skipped: int = Field(ge=0)

    total_chunks: int = Field(ge=0)

    minimum_chunk_tokens: int = Field(ge=0)
    maximum_chunk_tokens: int = Field(ge=0)
    average_chunk_tokens: float = Field(ge=0)

    generated_at_utc: datetime
