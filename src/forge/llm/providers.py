"""LLM and embedding provider backends (Ollama, OpenAI-compatible, Fake)."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import httpx

from forge.llm.models import EmbedResult, LLMMessage, LLMProviderError, LLMResponse


class LLMProvider(Protocol):
    """Protocol implemented by all chat and embedding providers."""

    name: str

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float,
        seed: int | None,
        json_mode: bool,
    ) -> LLMResponse: ...

    def embed(self, texts: list[str], *, model: str) -> EmbedResult: ...


class OllamaProvider:
    """HTTP provider for a local or remote Ollama server."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float,
        seed: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        try:
            with self._client() as client:
                resp = client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Ollama chat request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        content = str(data.get("message", {}).get("content", ""))
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        return LLMResponse(
            content=content,
            model=str(data.get("model") or model),
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency_ms, 2),
            cached=False,
            raw=data,
        )

    def embed(self, texts: list[str], *, model: str) -> EmbedResult:
        if not texts:
            return EmbedResult(embeddings=[], model=model, provider=self.name)

        start = time.perf_counter()
        try:
            with self._client() as client:
                resp = client.post("/api/embed", json={"model": model, "input": texts})
                if resp.status_code == 404:
                    # Fallback for older Ollama servers supporting only /api/embeddings
                    vectors: list[list[float]] = []
                    for text in texts:
                        legacy = client.post(
                            "/api/embeddings",
                            json={"model": model, "prompt": text},
                        )
                        legacy.raise_for_status()
                        vectors.append([float(x) for x in legacy.json()["embedding"]])
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return EmbedResult(
                        embeddings=vectors,
                        model=model,
                        provider=self.name,
                        latency_ms=round(latency_ms, 2),
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Ollama embed request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        embeddings = [[float(v) for v in vec] for vec in data.get("embeddings", [])]
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        return EmbedResult(
            embeddings=embeddings,
            model=str(data.get("model") or model),
            provider=self.name,
            prompt_tokens=prompt_tokens,
            latency_ms=round(latency_ms, 2),
        )


class OpenAICompatProvider:
    """HTTP provider for OpenAI-compatible chat and embedding endpoints."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise LLMProviderError("openai_compat_base_url must be configured for openai_compat")
        cleaned = base_url.rstrip("/")
        self.base_url = cleaned if cleaned.endswith("/v1") else f"{cleaned}/v1"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def _client(self) -> httpx.Client:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float,
        seed: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            with self._client() as client:
                resp = client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"OpenAI-compatible chat request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        choices = data.get("choices") or []
        content = str(choices[0].get("message", {}).get("content", "")) if choices else ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return LLMResponse(
            content=content,
            model=str(data.get("model") or model),
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            cached=False,
            raw=data,
        )

    def embed(self, texts: list[str], *, model: str) -> EmbedResult:
        if not texts:
            return EmbedResult(embeddings=[], model=model, provider=self.name)

        start = time.perf_counter()
        try:
            with self._client() as client:
                resp = client.post("/embeddings", json={"model": model, "input": texts})
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"OpenAI-compatible embed request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        items = sorted(data.get("data", []), key=lambda item: int(item.get("index", 0)))
        embeddings = [[float(v) for v in item.get("embedding", [])] for item in items]
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        return EmbedResult(
            embeddings=embeddings,
            model=str(data.get("model") or model),
            provider=self.name,
            prompt_tokens=prompt_tokens,
            latency_ms=round(latency_ms, 2),
        )


def deterministic_hash_embedding(text: str, dim: int = 64) -> list[float]:
    """Produce a deterministic L2-normalized bag-of-tokens embedding for tests."""
    vec = [0.0] * dim
    tokens = [tok for tok in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()]
    if not tokens:
        vec[0] = 1.0
        return vec
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeProvider:
    """Scripted deterministic provider for unit tests without live models."""

    name = "fake"

    def __init__(
        self,
        responses: Iterable[str | Exception | Callable[[list[LLMMessage]], str]] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_dim: int = 64,
    ) -> None:
        self._responses: list[str | Exception | Callable[[list[LLMMessage]], str]] = list(
            responses or []
        )
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def queue_response(
        self, response: str | Exception | Callable[[list[LLMMessage]], str]
    ) -> None:
        """Append another scripted response to the queue."""
        self._responses.append(response)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float,
        seed: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        self.chat_calls.append(
            {
                "messages": [m.model_dump() for m in messages],
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "json_mode": json_mode,
            }
        )
        if not self._responses:
            raise LLMProviderError("FakeProvider has no scripted responses remaining.")

        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        content = item(messages) if callable(item) else item

        prompt_tokens = sum(max(1, len(m.content.split())) for m in messages)
        completion_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=1.0,
            cached=False,
        )

    def embed(self, texts: list[str], *, model: str) -> EmbedResult:
        self.embed_calls.append({"texts": list(texts), "model": model})
        if self._embed_fn is not None:
            vectors = self._embed_fn(texts)
        else:
            vectors = [deterministic_hash_embedding(t, dim=self._embed_dim) for t in texts]
        prompt_tokens = sum(max(1, len(t.split())) for t in texts)
        return EmbedResult(
            embeddings=vectors,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            latency_ms=0.5,
        )
