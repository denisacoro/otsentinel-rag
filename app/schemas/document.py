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
