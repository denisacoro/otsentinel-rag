from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Protocol

from pydantic import ValidationError

from app.schemas.document import (
    ChunkingSummary,
    DocumentChunk,
    StructuredSection,
)


class TokenizerLike(Protocol):
    """Minimal tokenizer interface required by the chunker."""

    model_max_length: int

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Encode text into token IDs."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """Decode token IDs into text."""


def calculate_text_sha256(text: str) -> str:
    """Calculate a stable checksum for text content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_sections_jsonl(
    input_path: Path,
) -> list[StructuredSection]:
    """Load and validate structure-aware sections from JSONL."""

    if not input_path.exists():
        raise FileNotFoundError(f"Sections file does not exist: {input_path}")

    sections: list[StructuredSection] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                section = StructuredSection.model_validate_json(stripped_line)
            except ValidationError as error:
                raise ValueError(f"Invalid section on line {line_number}: {error}") from error

            sections.append(section)

    if not sections:
        raise ValueError(f"No valid sections were found in: {input_path}")

    return sections


def create_embedding_prefix(
    *,
    title: str,
    heading: str | None,
    heading_path: list[str],
) -> str:
    """Create contextual metadata prepended for embedding."""

    prefix_parts = [f"Document: {title}"]

    if heading_path:
        prefix_parts.append("Section path: " + " > ".join(heading_path))
    elif heading:
        prefix_parts.append(f"Section: {heading}")

    return "\n".join(prefix_parts) + "\n\n"


def validate_chunking_parameters(
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    """Validate chunk size and overlap configuration."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def create_section_chunks(
    *,
    section: StructuredSection,
    title: str,
    tokenizer: TokenizerLike,
    tokenizer_name: str,
    config_name: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    """Create token-window chunks from one structured section."""

    validate_chunking_parameters(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not section.text.strip():
        return []

    prefix = create_embedding_prefix(
        title=title,
        heading=section.heading,
        heading_path=section.heading_path,
    )

    prefix_token_ids = tokenizer.encode(
        prefix,
        add_special_tokens=False,
    )

    available_body_tokens = max_tokens - len(prefix_token_ids)

    if available_body_tokens <= 0:
        raise ValueError(
            "The document title and heading path use all available "
            f"tokens for section {section.section_id}. "
            "Increase max_tokens or shorten the metadata."
        )

    if overlap_tokens >= available_body_tokens:
        raise ValueError(
            "overlap_tokens must be smaller than the number of tokens available for section text."
        )

    section_token_ids = tokenizer.encode(
        section.text,
        add_special_tokens=False,
    )

    if not section_token_ids:
        return []

    windows: list[tuple[int, int, str, str, int]] = []

    token_start = 0

    while token_start < len(section_token_ids):
        token_end = min(
            token_start + available_body_tokens,
            len(section_token_ids),
        )

        body_token_ids = section_token_ids[token_start:token_end]

        chunk_text = tokenizer.decode(
            body_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        embedding_text = prefix + chunk_text

        actual_token_count = len(
            tokenizer.encode(
                embedding_text,
                add_special_tokens=False,
            )
        )

        # Some tokenizers may produce a slightly different number of
        # tokens after decoding and re-encoding. Shrink the window when
        # necessary so the final embedding text stays below the limit.
        while actual_token_count > max_tokens and token_end > token_start + 1:
            token_end -= 1

            body_token_ids = section_token_ids[token_start:token_end]

            chunk_text = tokenizer.decode(
                body_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            embedding_text = prefix + chunk_text

            actual_token_count = len(
                tokenizer.encode(
                    embedding_text,
                    add_special_tokens=False,
                )
            )

        if not chunk_text:
            raise ValueError(f"Tokenizer produced an empty chunk for section {section.section_id}.")

        if actual_token_count > max_tokens:
            raise ValueError(
                f"Could not create a chunk within the token limit for section {section.section_id}."
            )

        windows.append(
            (
                token_start,
                token_end,
                chunk_text,
                embedding_text,
                actual_token_count,
            )
        )

        if token_end >= len(section_token_ids):
            break

        token_start = token_end - overlap_tokens

    chunk_count = len(windows)
    chunks: list[DocumentChunk] = []

    for window_index, window in enumerate(
        windows,
        start=1,
    ):
        (
            window_start,
            window_end,
            chunk_text,
            embedding_text,
            actual_token_count,
        ) = window

        overlap_with_previous = 0 if window_index == 1 else overlap_tokens

        chunk_id = f"{section.section_id}::{config_name}::chunk::{window_index:04d}"

        chunks.append(
            DocumentChunk(
                source_id=section.source_id,
                document_id=section.document_id,
                section_id=section.section_id,
                chunk_id=chunk_id,
                config_name=config_name,
                title=title,
                heading=section.heading,
                heading_level=section.heading_level,
                heading_path=section.heading_path,
                page_start=section.page_start,
                page_end=section.page_end,
                chunk_index=window_index,
                chunk_count=chunk_count,
                text=chunk_text,
                embedding_text=embedding_text,
                body_token_count=window_end - window_start,
                token_count=actual_token_count,
                token_start=window_start,
                token_end=window_end,
                overlap_with_previous=(overlap_with_previous),
                tokenizer_name=tokenizer_name,
                content_sha256=calculate_text_sha256(embedding_text),
            )
        )

    return chunks


def build_document_chunks(
    *,
    source_id: str,
    document_id: str,
    title: str,
    sections: list[StructuredSection],
    sections_path: Path,
    tokenizer: TokenizerLike,
    tokenizer_name: str,
    config_name: str,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[DocumentChunk], ChunkingSummary]:
    """Create retrieval chunks for all document sections."""

    validate_chunking_parameters(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    all_chunks: list[DocumentChunk] = []
    sections_with_content = 0
    empty_sections_skipped = 0

    for section in sections:
        section_chunks = create_section_chunks(
            section=section,
            title=title,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            config_name=config_name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        if section_chunks:
            sections_with_content += 1
            all_chunks.extend(section_chunks)
        else:
            empty_sections_skipped += 1

    token_counts = [chunk.token_count for chunk in all_chunks]

    summary = ChunkingSummary(
        source_id=source_id,
        document_id=document_id,
        title=title,
        config_name=config_name,
        input_sections_path=str(sections_path),
        tokenizer_name=tokenizer_name,
        strategy="section_token_window_v1",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        total_sections=len(sections),
        sections_with_content=sections_with_content,
        empty_sections_skipped=empty_sections_skipped,
        total_chunks=len(all_chunks),
        minimum_chunk_tokens=(min(token_counts) if token_counts else 0),
        maximum_chunk_tokens=(max(token_counts) if token_counts else 0),
        average_chunk_tokens=(mean(token_counts) if token_counts else 0.0),
        generated_at_utc=datetime.now(UTC),
    )

    return all_chunks, summary


def save_chunks_jsonl(
    chunks: list[DocumentChunk],
    output_path: Path,
) -> None:
    """Save retrieval chunks as JSONL."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for chunk in chunks:
                serialized = chunk.model_dump(mode="json")

                output_file.write(
                    json.dumps(
                        serialized,
                        ensure_ascii=False,
                    )
                )
                output_file.write("\n")

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_chunking_summary(
    summary: ChunkingSummary,
    output_path: Path,
) -> None:
    """Save the chunking summary as formatted JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
