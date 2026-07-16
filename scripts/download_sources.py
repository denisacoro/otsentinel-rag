import argparse
from pathlib import Path

from app.ingestion.downloader import download_manifest_sources

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "sources.jsonl"
)

DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "download_reports"
    / "latest.jsonl"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download authoritative OTSentinel AI sources."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again even when they already exist.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    results = download_manifest_sources(
        manifest_path=DEFAULT_MANIFEST,
        raw_directory=DEFAULT_RAW_DIRECTORY,
        report_path=DEFAULT_REPORT,
        force=arguments.force,
    )

    print(f"\nCompleted successfully: {len(results)} source(s)")


if __name__ == "__main__":
    main()