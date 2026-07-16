import hashlib
from pathlib import Path

from app.ingestion.downloader import calculate_sha256, load_manifest


def test_source_manifest_is_valid() -> None:
    manifest_path = Path("data/manifests/sources.jsonl")

    sources = load_manifest(manifest_path)

    assert len(sources) >= 1
    assert sources[0].source_id == "nist-sp-800-82-r3"
    assert sources[0].publisher == "NIST"
    assert sources[0].filename.endswith(".pdf")


def test_calculate_sha256(tmp_path: Path) -> None:
    test_content = b"OTSentinel AI"
    test_file = tmp_path / "example.txt"
    test_file.write_bytes(test_content)

    expected_checksum = hashlib.sha256(test_content).hexdigest()

    assert calculate_sha256(test_file) == expected_checksum