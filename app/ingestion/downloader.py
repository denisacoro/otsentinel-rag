from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from app.schemas.source import DownloadResult, SourceRecord

DEFAULT_HEADERS = {"User-Agent": ("OTSentinel-AI/0.1 (educational AI engineering project)")}


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> list[SourceRecord]:
    """Load and validate source records from a JSONL manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {manifest_path}")

    sources: list[SourceRecord] = []

    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                source = SourceRecord.model_validate_json(stripped_line)
            except ValidationError as error:
                raise ValueError(
                    f"Invalid source on manifest line {line_number}: {error}"
                ) from error

            sources.append(source)

    if not sources:
        raise ValueError("The source manifest contains no valid records.")

    return sources


def download_source(
    source: SourceRecord,
    raw_directory: Path,
    *,
    force: bool = False,
) -> DownloadResult:
    """Download one source and return its local metadata."""

    publisher_directory = raw_directory / source.publisher.lower()
    publisher_directory.mkdir(parents=True, exist_ok=True)

    destination = publisher_directory / source.filename

    if destination.exists() and not force:
        return DownloadResult(
            source_id=source.source_id,
            local_path=str(destination),
            sha256=calculate_sha256(destination),
            size_bytes=destination.stat().st_size,
            status="existing",
            checked_at_utc=datetime.now(UTC),
        )

    temporary_destination = destination.with_suffix(destination.suffix + ".part")

    try:
        with httpx.stream(
            method="GET",
            url=str(source.download_url),
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=120.0,
        ) as response:
            response.raise_for_status()

            with temporary_destination.open("wb") as output_file:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    output_file.write(chunk)

        temporary_destination.replace(destination)

    except Exception:
        temporary_destination.unlink(missing_ok=True)
        raise

    return DownloadResult(
        source_id=source.source_id,
        local_path=str(destination),
        sha256=calculate_sha256(destination),
        size_bytes=destination.stat().st_size,
        status="downloaded",
        checked_at_utc=datetime.now(UTC),
    )


def save_download_report(
    results: list[DownloadResult],
    report_path: Path,
) -> None:
    """Save download results as JSONL."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report:
        for result in results:
            serialized = result.model_dump(mode="json")
            report.write(json.dumps(serialized) + "\n")


def download_manifest_sources(
    manifest_path: Path,
    raw_directory: Path,
    report_path: Path,
    *,
    force: bool = False,
) -> list[DownloadResult]:
    """Download every enabled source from the manifest."""

    sources = load_manifest(manifest_path)
    results: list[DownloadResult] = []

    for source in sources:
        if not source.enabled:
            continue

        print(f"Processing: {source.source_id}")

        result = download_source(
            source=source,
            raw_directory=raw_directory,
            force=force,
        )

        results.append(result)

        print(f"  Status: {result.status}\n  File: {result.local_path}\n  SHA-256: {result.sha256}")

    save_download_report(results, report_path)

    return results
