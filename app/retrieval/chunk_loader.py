from pathlib import Path

from pydantic import ValidationError

from app.schemas.document import DocumentChunk


def load_chunks_jsonl(
    input_path: Path,
) -> list[DocumentChunk]:
    """Load and validate retrieval chunks from JSONL."""

    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {input_path}")

    chunks: list[DocumentChunk] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                chunk = DocumentChunk.model_validate_json(stripped_line)
            except ValidationError as error:
                raise ValueError(f"Invalid chunk on line {line_number}: {error}") from error

            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No valid chunks were found in: {input_path}")

    return chunks
