from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceRecord(BaseModel):
    """Definition of an authoritative source used by OTSentinel AI."""

    source_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    title: str
    publisher: str
    document_type: str
    language: str = Field(min_length=2, max_length=5)
    version: str
    published_at: str

    source_page_url: HttpUrl
    download_url: HttpUrl

    filename: str
    license_notes: str
    enabled: bool = True


class DownloadResult(BaseModel):
    """Information produced after downloading or verifying a source."""

    source_id: str
    local_path: str
    sha256: str
    size_bytes: int
    status: Literal["downloaded", "existing"]
    checked_at_utc: datetime