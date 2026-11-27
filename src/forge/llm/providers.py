"""LLM and embedding provider backends (Ollama, OpenAI-compatible, Fake)."""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import httpx

from forge.llm.models import EmbedResult, LLMMessage, LLMProviderError, LLMResponse


def _redact_secrets(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret and len(secret) >= 4:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


class _RateLimiter:
    """Thread-safe requests-per-minute pacer and max-concurrency gate."""

    def __init__(
        self,
        requests_per_minute: float | None = None,
        max_concurrency: int = 2,
    ) -> None:
        self.min_interval = (
            60.0 / requests_per_minute
            if requests_per_minute is not None and requests_per_minute > 0
            else 0.0
        )
        self._semaphore = threading.Semaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._semaphore.acquire()
        try:
            if self.min_interval > 0.0:
                with self._lock:
                    now = time.monotonic()
                    wait = self._next_allowed - now
                    if wait > 0:
                        time.sleep(wait)
                        now = time.monotonic()
                    self._next_allowed = now + self.min_interval
            yield
        finally:
            self._semaphore.release()


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
        max_http_retries: int = 2,
        backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_http_retries = max_http_retries
        self.backoff_seconds = backoff_seconds
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    def _post_with_backoff(
        self, client: httpx.Client, path: str, payload: dict[str, Any], *, model: str
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_http_retries + 1):
            try:
                resp = client.post(path, json=payload)
                if resp.status_code in {429, 502, 503, 504} and attempt < self.max_http_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                return resp
            except httpx.ConnectError as exc:
                raise LLMProviderError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Ensure Ollama is running ('ollama serve') and pull '{model}' "
                    f"('ollama pull {model}'), or configure an OpenAI-compatible "
                    "endpoint in forge.toml / .env."
                ) from exc
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_http_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise LLMProviderError(
                    f"Ollama request to {path} timed out after {self.timeout_seconds}s "
                    f"({exc}). Try increasing timeout_seconds in forge.toml or using --fast."
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(f"Ollama request failed: {exc}") from exc
        raise LLMProviderError(f"Ollama request failed after retries: {last_exc}")

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
                resp = self._post_with_backoff(client, "/api/chat", payload, model=model)
                resp.raise_for_status()
                data = resp.json()
        except LLMProviderError:
            raise
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
                resp = self._post_with_backoff(
                    client, "/api/embed", {"model": model, "input": texts}, model=model
                )
                if resp.status_code == 404:
                    # Fallback for older Ollama servers supporting only /api/embeddings
                    vectors: list[list[float]] = []
                    for text in texts:
                        legacy = self._post_with_backoff(
                            client,
                            "/api/embeddings",
                            {"model": model, "prompt": text},
                            model=model,
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
        except LLMProviderError:
            raise
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
        *,
        append_v1: bool = True,
        include_seed: bool = True,
        require_api_key: bool = False,
        api_key_env: str = "FORGE_OPENAI_API_KEY",
        requests_per_minute: float | None = None,
        max_concurrency: int = 2,
        max_http_retries: int = 3,
        backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise LLMProviderError("openai_compat_base_url must be configured for openai_compat")
        cleaned = base_url.rstrip("/")
        if append_v1 and not cleaned.endswith("/v1"):
            self.base_url = f"{cleaned}/v1"
        else:
            self.base_url = cleaned
        self.api_key = api_key
        self.include_seed = include_seed
        self.require_api_key = require_api_key
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_http_retries = max_http_retries
        self.backoff_seconds = backoff_seconds
        self._limiter = _RateLimiter(
            requests_per_minute=requests_per_minute,
            max_concurrency=max_concurrency,
        )
        self._transport = transport

    def _ensure_api_key(self) -> None:
        if self.require_api_key and not self.api_key:
            raise LLMProviderError(
                f"Missing API key for provider '{self.name}': set {self.api_key_env} in "
                "your environment or gitignored .env file."
            )

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

    def _retry_delay(self, attempt: int, resp: httpx.Response | None = None) -> float:
        if resp is not None:
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                try:
                    return min(30.0, max(0.05, float(retry_after)))
                except ValueError:
                    pass
        base = max(0.01, self.backoff_seconds) * (2**attempt)
        jitter = random.uniform(0.8, 1.2)
        return min(30.0, base * jitter)

    def _post_with_backoff(
        self, client: httpx.Client, path: str, payload: dict[str, Any]
    ) -> httpx.Response:
        last_exc: Exception | None = None
        last_resp: httpx.Response | None = None

        for attempt in range(self.max_http_retries + 1):
            try:
                with self._limiter.acquire():
                    resp = client.post(path, json=payload)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_http_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise LLMProviderError(
                    f"{self.name} request to {path} timed out after {self.timeout_seconds}s: "
                    f"{_redact_secrets(str(exc), [self.api_key])}"
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError(
                    f"{self.name} HTTP error: {_redact_secrets(str(exc), [self.api_key])}"
                ) from exc

            if resp.status_code in (401, 403):
                raise LLMProviderError(
                    f"Authentication failed for provider '{self.name}' (HTTP {resp.status_code}). "
                    f"Verify {self.api_key_env} in your environment or .env."
                )

            if resp.status_code in {429, 500, 502, 503, 504}:
                last_resp = resp
                if attempt < self.max_http_retries:
                    time.sleep(self._retry_delay(attempt, resp))
                    continue
                if resp.status_code == 429:
                    raise LLMProviderError(
                        f"Rate limit / quota exhausted for provider '{self.name}' (HTTP 429) "
                        "after retries. Lower requests_per_minute in config, reduce eval size "
                        "with --fast, or wait for project quota reset."
                    )
                raise LLMProviderError(
                    f"Provider '{self.name}' failed with HTTP {resp.status_code} after retries."
                )

            if resp.status_code >= 400:
                detail = _redact_secrets(resp.text[:300], [self.api_key])
                raise LLMProviderError(
                    f"Provider '{self.name}' rejected request (HTTP {resp.status_code}): {detail}"
                )
            return resp

        if last_resp is not None and last_resp.status_code == 429:
            raise LLMProviderError(
                f"Rate limit / quota exhausted for provider '{self.name}' (HTTP 429)."
            )
        raise LLMProviderError(
            f"Provider '{self.name}' failed after retries: "
            f"{_redact_secrets(str(last_exc), [self.api_key])}"
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
        self._ensure_api_key()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if seed is not None and self.include_seed:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            with self._client() as client:
                resp = self._post_with_backoff(client, "/chat/completions", payload)
                data = resp.json()
        except LLMProviderError:
            raise
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"{self.name} chat request failed: "
                f"{_redact_secrets(str(exc), [self.api_key])}"
            ) from exc

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
        self._ensure_api_key()

        start = time.perf_counter()
        try:
            with self._client() as client:
                resp = self._post_with_backoff(
                    client, "/embeddings", {"model": model, "input": texts}
                )
                data = resp.json()
        except LLMProviderError:
            raise
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"{self.name} embed request failed: "
                f"{_redact_secrets(str(exc), [self.api_key])}"
            ) from exc

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


class GeminiProvider(OpenAICompatProvider):
    """Gemini API provider using Google's OpenAI-compatible v1beta/openai endpoint."""

    name = "gemini"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout_seconds: float = 120.0,
        *,
        requests_per_minute: float | None = 10.0,
        max_concurrency: int = 1,
        max_http_retries: int = 4,
        backoff_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            append_v1=False,
            include_seed=False,
            require_api_key=True,
            api_key_env="FORGE_GEMINI_API_KEY",
            requests_per_minute=requests_per_minute,
            max_concurrency=max_concurrency,
            max_http_retries=max_http_retries,
            backoff_seconds=backoff_seconds,
            transport=transport,
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
