from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pymupdf
from pydantic import BaseModel

from app.schemas.document import (
    ExtractedLine,
    StructuredSection,
    StructureSummary,
)

WHITESPACE_PATTERN = re.compile(r"\s+")

PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+)?\d+(?:\s+(?:of|/)\s+\d+)?$",
    re.IGNORECASE,
)

ROMAN_PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+)?[ivxlcdm]+$",
    re.IGNORECASE,
)

NUMBERED_HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+){0,5})\.?\s+\S")

APPENDIX_HEADING_PATTERN = re.compile(
    r"^appendix\s+[a-z]\b",
    re.IGNORECASE,
)

CAPTION_PATTERN = re.compile(
    r"^(?:figure|table|exhibit)\s+[a-z0-9.-]+",
    re.IGNORECASE,
)

BULLET_PATTERN = re.compile(r"^[•●▪◦\-–—]\s+")

KNOWN_STANDALONE_HEADINGS = {
    "abstract",
    "acknowledgements",
    "conclusion",
    "conclusions",
    "executive summary",
    "foreword",
    "glossary",
    "introduction",
    "references",
    "summary",
}


def normalize_visible_line(text: str) -> str:
    """Normalize one visual line while preserving its content."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00ad", "")
    normalized = normalized.replace("\u00a0", " ")

    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def create_margin_pattern(text: str) -> str:
    """Create a normalized key for repeated margin detection."""

    normalized = normalize_visible_line(text).casefold()

    if PAGE_NUMBER_PATTERN.fullmatch(normalized):
        return "<page-number>"

    if ROMAN_PAGE_NUMBER_PATTERN.fullmatch(normalized):
        return "<page-number>"

    # Normalize changing standalone numbers, such as "Page 4 of 316".
    normalized = re.sub(r"\b\d+\b", "<number>", normalized)

    return normalized


def span_is_bold(span: dict) -> bool:
    """Determine whether a PyMuPDF text span is bold."""

    flags = int(span.get("flags", 0))
    font_name = str(span.get("font", "")).casefold()

    bold_flag = bool(flags & (1 << 4))

    bold_name = any(
        marker in font_name
        for marker in (
            "bold",
            "black",
            "demi",
            "semibold",
        )
    )

    return bold_flag or bold_name


def join_span_text(spans: list[dict]) -> str:
    """Combine span text belonging to the same visual line."""

    return normalize_visible_line("".join(str(span.get("text", "")) for span in spans))


def calculate_weighted_font_size(spans: list[dict]) -> float:
    """Calculate a character-weighted font size for one line."""

    weighted_total = 0.0
    character_total = 0

    for span in spans:
        span_text = str(span.get("text", "")).strip()
        character_count = max(len(span_text), 1)
        span_size = float(span.get("size", 0.0))

        weighted_total += span_size * character_count
        character_total += character_count

    if character_total == 0:
        return 1.0

    return weighted_total / character_total


def calculate_line_boldness(spans: list[dict]) -> bool:
    """Mark a line as bold when at least half its text is bold."""

    total_characters = 0
    bold_characters = 0

    for span in spans:
        character_count = max(
            len(str(span.get("text", "")).strip()),
            1,
        )

        total_characters += character_count

        if span_is_bold(span):
            bold_characters += character_count

    if total_characters == 0:
        return False

    return bold_characters / total_characters >= 0.5


def classify_page_position(
    *,
    bbox: tuple[float, float, float, float],
    page_height: float,
    header_ratio: float,
    footer_ratio: float,
) -> str:
    """Classify a line as header, body or footer by page position."""

    _, y0, _, y1 = bbox

    if y1 <= page_height * header_ratio:
        return "header"

    if y0 >= page_height * (1.0 - footer_ratio):
        return "footer"

    return "body"


def extract_layout_lines(
    *,
    source_id: str,
    document_id: str,
    pdf_path: Path,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
) -> tuple[list[ExtractedLine], int]:
    """Extract layout-aware lines from every page."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    lines: list[ExtractedLine] = []

    with pymupdf.open(pdf_path) as document:
        total_pages = document.page_count

        for page_index in range(total_pages):
            page = document.load_page(page_index)

            page_dictionary = page.get_text(
                "dict",
                sort=True,
            )

            page_width = float(page.rect.width)
            page_height = float(page.rect.height)

            for block in page_dictionary.get("blocks", []):
                block_lines = block.get("lines")

                if not block_lines:
                    continue

                for line in block_lines:
                    spans = [
                        span for span in line.get("spans", []) if str(span.get("text", "")).strip()
                    ]

                    if not spans:
                        continue

                    text = join_span_text(spans)

                    if not text:
                        continue

                    raw_bbox = line.get("bbox")

                    if raw_bbox is None or len(raw_bbox) != 4:
                        continue

                    bbox = (
                        float(raw_bbox[0]),
                        float(raw_bbox[1]),
                        float(raw_bbox[2]),
                        float(raw_bbox[3]),
                    )

                    position = classify_page_position(
                        bbox=bbox,
                        page_height=page_height,
                        header_ratio=header_ratio,
                        footer_ratio=footer_ratio,
                    )

                    font_names = sorted({str(span.get("font", "unknown")) for span in spans})

                    lines.append(
                        ExtractedLine(
                            source_id=source_id,
                            document_id=document_id,
                            page_number=page_index + 1,
                            text=text,
                            bbox=bbox,
                            page_width=page_width,
                            page_height=page_height,
                            font_size=calculate_weighted_font_size(spans),
                            font_names=font_names,
                            is_bold=calculate_line_boldness(spans),
                            position=position,
                        )
                    )

    return lines, total_pages


def detect_repeated_margin_patterns(
    lines: list[ExtractedLine],
    *,
    total_pages: int,
    minimum_page_fraction: float = 0.5,
    minimum_pages: int = 3,
) -> set[tuple[str, str]]:
    """Find header and footer lines repeated across many pages."""

    occurrences: dict[
        tuple[str, str],
        set[int],
    ] = defaultdict(set)

    for line in lines:
        if line.position == "body":
            continue

        pattern = create_margin_pattern(line.text)

        if len(pattern) < 2 or len(pattern) > 160:
            continue

        key = (line.position, pattern)
        occurrences[key].add(line.page_number)

    required_pages = max(
        minimum_pages,
        math.ceil(total_pages * minimum_page_fraction),
    )

    return {key for key, page_numbers in occurrences.items() if len(page_numbers) >= required_pages}


def mark_repeated_margin_lines(
    lines: list[ExtractedLine],
    repeated_patterns: set[tuple[str, str]],
) -> list[ExtractedLine]:
    """Mark lines that match detected repeated margin patterns."""

    marked_lines: list[ExtractedLine] = []

    for line in lines:
        pattern_key = (
            line.position,
            create_margin_pattern(line.text),
        )

        marked_lines.append(
            line.model_copy(update={"is_repeated_margin": (pattern_key in repeated_patterns)})
        )

    return marked_lines


def estimate_body_font_size(
    lines: list[ExtractedLine],
) -> float:
    """Estimate the dominant body font using weighted frequency."""

    weighted_sizes: Counter[float] = Counter()

    for line in lines:
        if line.position != "body":
            continue

        if line.is_repeated_margin:
            continue

        rounded_size = round(line.font_size * 2.0) / 2.0
        weighted_sizes[rounded_size] += max(len(line.text), 1)

    if not weighted_sizes:
        raise ValueError("Could not estimate body font size.")

    return weighted_sizes.most_common(1)[0][0]


def numbered_heading_level(text: str) -> int | None:
    """Return hierarchy level from a numbered heading."""

    match = NUMBERED_HEADING_PATTERN.match(text)

    if match is None:
        return None

    number = match.group("number")

    return min(number.count(".") + 1, 6)


def looks_like_title(text: str) -> bool:
    """Determine whether a short line resembles a title."""

    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if word]

    if not words:
        return False

    capitalized_words = sum(word[0].isupper() for word in words)

    return capitalized_words / len(words) >= 0.6


def detect_heading_level(
    line: ExtractedLine,
    body_font_size: float,
) -> int:
    """Determine an approximate hierarchy level for a heading."""

    numbered_level = numbered_heading_level(line.text)

    if numbered_level is not None:
        return numbered_level

    if APPENDIX_HEADING_PATTERN.match(line.text):
        return 1

    size_ratio = line.font_size / body_font_size

    if size_ratio >= 1.45:
        return 1

    if size_ratio >= 1.25:
        return 2

    if size_ratio >= 1.10:
        return 3

    return 3


def line_is_heading(
    line: ExtractedLine,
    body_font_size: float,
) -> bool:
    """Determine whether a layout line is probably a heading."""

    if line.position != "body":
        return False

    if line.is_repeated_margin:
        return False

    text = line.text.strip()
    word_count = len(text.split())

    if not text:
        return False

    if len(text) > 180 or word_count > 24:
        return False

    if BULLET_PATTERN.match(text):
        return False

    if CAPTION_PATTERN.match(text):
        return False

    size_ratio = line.font_size / body_font_size
    numbered_level = numbered_heading_level(text)

    # Numbered list items should not become headings unless typography
    # supports the classification.
    if numbered_level is not None:
        return line.is_bold or size_ratio >= 1.05

    if APPENDIX_HEADING_PATTERN.match(text):
        return line.is_bold or size_ratio >= 1.10

    if text.casefold() in KNOWN_STANDALONE_HEADINGS:
        return line.is_bold or size_ratio >= 1.05

    if size_ratio >= 1.25:
        return True

    if (
        line.is_bold
        and size_ratio >= 1.02
        and word_count <= 16
        and not text.endswith((".", ",", ";"))
    ):
        return True

    if text.isupper() and word_count <= 12 and size_ratio >= 1.05:
        return True

    return line.is_bold and looks_like_title(text) and word_count <= 12 and not text.endswith(".")


def classify_heading_lines(
    lines: list[ExtractedLine],
    body_font_size: float,
) -> list[ExtractedLine]:
    """Add heading labels and hierarchy levels to lines."""

    classified_lines: list[ExtractedLine] = []

    for line in lines:
        is_heading = line_is_heading(
            line=line,
            body_font_size=body_font_size,
        )

        heading_level = None

        if is_heading:
            heading_level = detect_heading_level(
                line=line,
                body_font_size=body_font_size,
            )

        classified_lines.append(
            line.model_copy(
                update={
                    "is_heading": is_heading,
                    "heading_level": heading_level,
                }
            )
        )

    return classified_lines


def create_section(
    *,
    source_id: str,
    document_id: str,
    section_index: int,
    heading: str | None,
    heading_level: int,
    heading_path: list[str],
    content_lines: list[ExtractedLine],
) -> StructuredSection:
    """Create one validated section from accumulated lines."""

    section_text = "\n".join(line.text for line in content_lines).strip()

    return StructuredSection(
        source_id=source_id,
        document_id=document_id,
        section_id=(f"{document_id}::section::{section_index:04d}"),
        heading=heading,
        heading_level=heading_level,
        heading_path=heading_path,
        page_start=content_lines[0].page_number,
        page_end=content_lines[-1].page_number,
        text=section_text,
        character_count=len(section_text),
        word_count=len(section_text.split()),
    )


def build_sections(
    *,
    source_id: str,
    document_id: str,
    lines: list[ExtractedLine],
) -> list[StructuredSection]:
    """Build hierarchical sections from classified layout lines."""

    sections: list[StructuredSection] = []
    heading_by_level: dict[int, str] = {}

    current_heading: str | None = None
    current_heading_level = 0
    current_heading_path: list[str] = []
    current_content: list[ExtractedLine] = []

    def flush_current_section() -> None:
        nonlocal current_content

        if not current_content:
            return

        sections.append(
            create_section(
                source_id=source_id,
                document_id=document_id,
                section_index=len(sections) + 1,
                heading=current_heading,
                heading_level=current_heading_level,
                heading_path=current_heading_path.copy(),
                content_lines=current_content,
            )
        )

        current_content = []

    for line in lines:
        if line.is_repeated_margin:
            continue

        if line.is_heading:
            flush_current_section()

            heading_level = line.heading_level or 1

            obsolete_levels = [level for level in heading_by_level if level >= heading_level]

            for level in obsolete_levels:
                del heading_by_level[level]

            heading_by_level[heading_level] = line.text

            current_heading = line.text
            current_heading_level = heading_level
            current_heading_path = [
                heading_by_level[level]
                for level in sorted(heading_by_level)
                if level <= heading_level
            ]

            continue

        current_content.append(line)

    flush_current_section()

    return sections


def extract_document_structure(
    *,
    source_id: str,
    document_id: str,
    title: str,
    pdf_path: Path,
) -> tuple[
    list[StructuredSection],
    StructureSummary,
    list[ExtractedLine],
]:
    """Run the complete structure extraction pipeline."""

    raw_lines, total_pages = extract_layout_lines(
        source_id=source_id,
        document_id=document_id,
        pdf_path=pdf_path,
    )

    repeated_patterns = detect_repeated_margin_patterns(
        raw_lines,
        total_pages=total_pages,
    )

    marked_lines = mark_repeated_margin_lines(
        raw_lines,
        repeated_patterns,
    )

    body_font_size = estimate_body_font_size(marked_lines)

    classified_lines = classify_heading_lines(
        marked_lines,
        body_font_size,
    )

    sections = build_sections(
        source_id=source_id,
        document_id=document_id,
        lines=classified_lines,
    )

    repeated_headers = sorted(
        pattern for position, pattern in repeated_patterns if position == "header"
    )

    repeated_footers = sorted(
        pattern for position, pattern in repeated_patterns if position == "footer"
    )

    summary = StructureSummary(
        source_id=source_id,
        document_id=document_id,
        title=title,
        input_path=str(pdf_path),
        total_pages=total_pages,
        total_lines=len(classified_lines),
        repeated_margin_lines=sum(line.is_repeated_margin for line in classified_lines),
        detected_headings=sum(line.is_heading for line in classified_lines),
        generated_sections=len(sections),
        body_font_size=body_font_size,
        repeated_header_patterns=repeated_headers,
        repeated_footer_patterns=repeated_footers,
        processed_at_utc=datetime.now(UTC),
    )

    return sections, summary, classified_lines


def save_models_jsonl(
    records: Iterable[BaseModel],
    output_path: Path,
) -> None:
    """Save Pydantic models as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for record in records:
                serialized = record.model_dump(mode="json")

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


def save_structure_summary(
    summary: StructureSummary,
    output_path: Path,
) -> None:
    """Save the structure extraction summary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
