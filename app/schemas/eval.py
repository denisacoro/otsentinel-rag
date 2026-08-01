from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EvalQuestion(BaseModel):
    """One golden evaluation question."""

    question_id: str
    question: str
    language: Literal["en", "ro"]

    reference_answer: str

    relevant_document_ids: list[str]
    relevant_chunk_ids: list[str]

    question_type: Literal[
        "concept_explanation",
        "mitigation_recommendation",
        "protocol_security_explanation",
        "advisory_summary",
        "affected_product_lookup",
        "attack_mapping",
        "multi_document",
        "refusal_unsupported",
    ]

    answerable: bool
    split: Literal["validation", "test"]

    note: str | None = None