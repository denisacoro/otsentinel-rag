from __future__ import annotations

import re

EXCESSIVE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def normalize_extracted_text(text: str) -> str:
    """Apply safe basic normalization to extracted PDF text."""

    normalized = text.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")

    # Remove soft hyphen characters that may appear during PDF extraction.
    normalized = normalized.replace("\u00ad", "")

    # Replace non-breaking spaces with normal spaces.
    normalized = normalized.replace("\u00a0", " ")

    # Remove trailing spaces while preserving useful indentation.
    lines = [line.rstrip() for line in normalized.splitlines()]
    normalized = "\n".join(lines)

    # Avoid very large empty regions in the extracted text.
    normalized = EXCESSIVE_BLANK_LINES_PATTERN.sub("\n\n", normalized)

    return normalized.strip()
