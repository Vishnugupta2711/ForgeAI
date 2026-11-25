"""Data models for ForgeAI groundability verification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

GroundabilityStatus = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]


class StrictGroundabilityError(RuntimeError):
    """Raised in --strict mode when any topic has PARTIAL or UNSUPPORTED status."""

    def __init__(
        self,
        message: str,
        *,
        report: Any = None,
        requirements: Any = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.requirements = requirements


class AtomicTopic(BaseModel):
    """Single decomposed topic or implicit capability requirement to verify."""

    topic_id: str
    title: str
    description: str
    source_req_id: str | None = None
    is_implicit: bool = False
    capability_hint: str | None = None


class EvidenceQuote(BaseModel):
    """Quote returned by the LLM groundability check with substring verification status."""

    doc_id: str = ""
    chunk_id: str
    quote: str
    verified_substring: bool = False


class TopicGroundability(BaseModel):
    """Groundability verdict and deterministic evidence metrics for one topic."""

    topic_id: str
    title: str
    description: str
    source_req_id: str | None = None
    status: GroundabilityStatus
    coverage_score: float = Field(ge=0.0, le=1.0)
    key_term_overlap: float = Field(ge=0.0, le=1.0)
    max_retrieval_score: float = Field(ge=0.0)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    quotes: list[EvidenceQuote] = Field(default_factory=list)
    reasoning: str = ""
    gap_note: str | None = None
    quote_verification_failed: bool = False


class GroundabilityReport(BaseModel):
    """Aggregate groundability assessment across all decomposed topics."""

    strict_mode: bool = False
    has_gaps: bool = False
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    flagged_gaps: list[str] = Field(default_factory=list)
    topics: list[TopicGroundability] = Field(default_factory=list)
