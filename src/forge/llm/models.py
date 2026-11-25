"""Data models and exceptions for forge.llm."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class LLMProviderError(RuntimeError):
    """Raised when an LLM or embedding provider request fails."""


class LLMValidationError(ValueError):
    """Raised when LLM output fails JSON parsing or Pydantic validation after retries."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class LLMMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str


class LLMResponse(BaseModel):
    """Structured response returned by LLMClient.chat."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class EmbedResult(BaseModel):
    """Structured embedding response returned by a provider."""

    embeddings: list[list[float]]
    model: str
    provider: str
    prompt_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class UsageRecord(BaseModel):
    """Single usage event written to the run-level JSONL usage log."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    operation: str
    role: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    attempt: int = 1
    status: str = "ok"
    error: str | None = None
