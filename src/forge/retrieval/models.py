"""Data models for hybrid retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from forge.ingest.models import Chunk


class RetrievalConfig(BaseModel):
    """Configuration parameters for hybrid BM25 + dense RRF retrieval."""

    alpha: float = 0.5
    top_k: int = 5
    rrf_k: int = 60
    min_score: float | None = None
    query_rewrite: bool = False

    @field_validator("alpha")
    @classmethod
    def _validate_alpha(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        return value

    @field_validator("top_k", "rrf_k")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k and rrf_k must be >= 1.")
        return value


class SearchHit(BaseModel):
    """Ranked retrieval result with component scores and ranks."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)
