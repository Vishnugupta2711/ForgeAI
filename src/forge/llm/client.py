"""Unified LLMClient with JSON schema repair-retry, disk cache, and usage logging."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from forge.config import ForgeConfig, get_gemini_api_key, load_config, load_prompt
from forge.llm.models import (
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMValidationError,
    UsageRecord,
)
from forge.llm.providers import (
    FakeProvider,
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenAICompatProvider,
)

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _normalize_messages(messages: str | Sequence[LLMMessage | dict[str, str]]) -> list[LLMMessage]:
    if isinstance(messages, str):
        return [LLMMessage(role="user", content=messages)]
    out: list[LLMMessage] = []
    for item in messages:
        if isinstance(item, LLMMessage):
            out.append(item)
        else:
            out.append(LLMMessage(role=item["role"], content=item["content"]))
    return out


def extract_json_value(text: str) -> Any:
    """Parse JSON from raw model text, handling code fences and surrounding prose."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty response cannot be parsed as JSON.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for match in _CODE_FENCE_RE.finditer(stripped):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch in "{[":
            try:
                parsed, _ = decoder.raw_decode(stripped[idx:])
                return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from response: {stripped[:200]}")


def _cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMClient:
    """Single boundary for all LLM and embedding calls across ForgeAI."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        providers_by_role: dict[str, LLMProvider] | None = None,
        config: ForgeConfig | None = None,
        cache_dir: Path | str | None = None,
        usage_log_path: Path | str | None = None,
        cache_enabled: bool = True,
        max_calls_per_run: int | None = None,
    ) -> None:
        self.provider = provider
        self.providers_by_role = dict(providers_by_role or {})
        self.config = config or ForgeConfig()
        self.cache_enabled = cache_enabled and (cache_dir is not None)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.usage_log_path = Path(usage_log_path) if usage_log_path is not None else None
        self.usage_records: list[UsageRecord] = []
        self.max_calls_per_run = (
            max_calls_per_run
            if max_calls_per_run is not None
            else self.config.budgets.max_calls_per_run
        )
        self._live_calls = 0

    @staticmethod
    def _build_provider(name: str, cfg: ForgeConfig) -> LLMProvider:
        rl = cfg.rate_limits
        if name == "ollama":
            return OllamaProvider(
                base_url=cfg.endpoints.ollama_base_url,
                timeout_seconds=cfg.endpoints.timeout_seconds,
                max_http_retries=rl.max_http_retries,
                backoff_seconds=rl.backoff_seconds,
            )
        if name == "openai_compat":
            return OpenAICompatProvider(
                base_url=cfg.endpoints.openai_compat_base_url,
                api_key=cfg.endpoints.openai_api_key,
                timeout_seconds=cfg.endpoints.timeout_seconds,
                requests_per_minute=rl.requests_per_minute,
                max_concurrency=rl.max_concurrency,
                max_http_retries=rl.max_http_retries,
                backoff_seconds=rl.backoff_seconds,
            )
        if name == "gemini":
            return GeminiProvider(
                api_key=get_gemini_api_key(),
                base_url=cfg.endpoints.gemini_base_url,
                timeout_seconds=cfg.endpoints.timeout_seconds,
                requests_per_minute=rl.requests_per_minute,
                max_concurrency=rl.max_concurrency,
                max_http_retries=rl.max_http_retries,
                backoff_seconds=rl.backoff_seconds,
            )
        raise LLMProviderError(f"Unsupported provider: {name}")

    @classmethod
    def from_config(
        cls,
        config: ForgeConfig | None = None,
        *,
        usage_log_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> LLMClient:
        """Construct an LLMClient from ForgeConfig with optional per-role routing."""
        cfg = config or load_config()
        pool: dict[str, LLMProvider] = {}
        default_provider = cls._build_provider(cfg.endpoints.provider, cfg)
        pool[cfg.endpoints.provider] = default_provider

        role_map: dict[str, LLMProvider] = {}
        for role_name in ("agent", "judge", "eval_writer", "embed"):
            pname = cfg.provider_for_role(role_name)
            if pname not in pool:
                pool[pname] = cls._build_provider(pname, cfg)
            role_map[role_name] = pool[pname]

        resolved_cache: Path | None = None
        if cfg.cache.enabled:
            resolved_cache = Path(cache_dir) if cache_dir is not None else Path(cfg.cache.dir)

        return cls(
            provider=default_provider,
            providers_by_role=role_map,
            config=cfg,
            cache_dir=resolved_cache,
            usage_log_path=usage_log_path,
            cache_enabled=cfg.cache.enabled,
            max_calls_per_run=cfg.budgets.max_calls_per_run,
        )

    def _provider_for_role(self, role: str) -> LLMProvider:
        return self.providers_by_role.get(role, self.provider)

    def _check_call_budget(self) -> None:
        if self.max_calls_per_run is not None and self._live_calls >= self.max_calls_per_run:
            raise LLMProviderError(
                f"Per-run LLM call budget exhausted ({self.max_calls_per_run} live calls). "
                "Increase budgets.max_calls_per_run or use --fast."
            )

    def _model_for_role(self, role: str) -> str:
        mapping = {
            "agent": self.config.models.agent,
            "judge": self.config.models.judge,
            "eval_writer": self.config.models.eval_writer,
            "embed": self.config.models.embed,
        }
        return mapping.get(role, self.config.models.agent)

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self.cache_enabled or self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        if not self.cache_enabled or self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _log_usage(self, record: UsageRecord) -> None:
        self.usage_records.append(record)
        if self.usage_log_path is not None:
            self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log_path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")

    def set_usage_log_path(self, path: Path | str | None) -> None:
        """Update the run-level JSONL usage log destination."""
        self.usage_log_path = Path(path) if path is not None else None

    def usage_summary(self) -> dict[str, Any]:
        """Return aggregate token, latency, and cache-hit metrics for this client."""
        return {
            "calls": len(self.usage_records),
            "cache_hits": sum(1 for r in self.usage_records if r.cached),
            "prompt_tokens": sum(r.prompt_tokens for r in self.usage_records),
            "completion_tokens": sum(r.completion_tokens for r in self.usage_records),
            "total_tokens": sum(r.total_tokens for r in self.usage_records),
            "total_latency_ms": round(sum(r.latency_ms for r in self.usage_records), 2),
        }

    def chat(
        self,
        messages: str | Sequence[LLMMessage | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
        role: str = "agent",
        attempt: int = 1,
    ) -> LLMResponse:
        """Send a chat completion request with disk caching and usage logging."""
        norm_messages = _normalize_messages(messages)
        resolved_model = model or self._model_for_role(role)
        active_provider = self._provider_for_role(role)
        resolved_temp = (
            self.config.generation.temperature if temperature is None else temperature
        )
        resolved_seed = self.config.generation.seed if seed is None else seed

        key = _cache_key(
            {
                "op": "chat",
                "provider": active_provider.name,
                "model": resolved_model,
                "messages": [m.model_dump() for m in norm_messages],
                "temperature": resolved_temp,
                "seed": resolved_seed,
                "json_mode": json_mode,
            }
        )

        cached_payload = self._read_cache(key)
        if cached_payload is not None:
            response = LLMResponse.model_validate(cached_payload).model_copy(
                update={"cached": True, "latency_ms": 0.0}
            )
            self._log_usage(
                UsageRecord(
                    operation="chat",
                    role=role,
                    provider=response.provider,
                    model=response.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    latency_ms=0.0,
                    cached=True,
                    attempt=attempt,
                    status="ok",
                )
            )
            return response

        self._check_call_budget()
        self._live_calls += 1
        try:
            response = active_provider.chat(
                norm_messages,
                model=resolved_model,
                temperature=resolved_temp,
                seed=resolved_seed,
                json_mode=json_mode,
            )
        except Exception as exc:
            self._log_usage(
                UsageRecord(
                    operation="chat",
                    role=role,
                    provider=active_provider.name,
                    model=resolved_model,
                    attempt=attempt,
                    status="error",
                    error=str(exc),
                )
            )
            raise

        self._write_cache(key, response.model_dump())
        self._log_usage(
            UsageRecord(
                operation="chat",
                role=role,
                provider=response.provider,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                cached=False,
                attempt=attempt,
                status="ok",
            )
        )
        return response

    def json(
        self,
        messages: str | Sequence[LLMMessage | dict[str, str]],
        *,
        schema: type[T],
        model: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_repairs: int = 2,
        role: str = "agent",
    ) -> T:
        """Call chat in JSON mode, validate against schema, and repair-retry on errors."""
        conversation = _normalize_messages(messages)
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        repair_template = load_prompt("llm_json_repair.v1.md")
        errors: list[str] = []

        for attempt_idx in range(1, max_repairs + 2):
            resp = self.chat(
                conversation,
                model=model,
                temperature=temperature,
                seed=seed,
                json_mode=True,
                role=role,
                attempt=attempt_idx,
            )
            try:
                parsed = extract_json_value(resp.content)
                return schema.model_validate(parsed)
            except (ValueError, ValidationError) as exc:
                err_text = str(exc)
                errors.append(f"Attempt {attempt_idx}: {err_text}")
                if attempt_idx > max_repairs:
                    break
                repair_msg = repair_template.format(
                    error_message=err_text,
                    schema_json=schema_json,
                )
                conversation = [
                    *conversation,
                    LLMMessage(role="assistant", content=resp.content),
                    LLMMessage(role="user", content=repair_msg),
                ]

        raise LLMValidationError(
            f"Failed to validate output for schema {schema.__name__} after "
            f"{len(errors)} attempts.",
            errors=errors,
        )

    def embed(
        self,
        texts: str | list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for one or more texts with disk caching and usage logging."""
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return []

        resolved_model = (model if model is not None else self.config.models.embed).strip()
        if not resolved_model or resolved_model.lower() in {"none", "bm25"}:
            raise LLMProviderError("No embedding model configured; falling back to BM25-only.")

        active_provider = self._provider_for_role("embed")
        key = _cache_key(
            {
                "op": "embed",
                "provider": active_provider.name,
                "model": resolved_model,
                "texts": batch,
            }
        )

        start = time.perf_counter()
        cached_payload = self._read_cache(key)
        if cached_payload is not None:
            vectors = cached_payload["embeddings"]
            self._log_usage(
                UsageRecord(
                    operation="embed",
                    role="embed",
                    provider=active_provider.name,
                    model=resolved_model,
                    prompt_tokens=int(cached_payload.get("prompt_tokens", 0)),
                    total_tokens=int(cached_payload.get("prompt_tokens", 0)),
                    latency_ms=0.0,
                    cached=True,
                    status="ok",
                )
            )
            return vectors

        self._check_call_budget()
        self._live_calls += 1
        try:
            result = active_provider.embed(batch, model=resolved_model)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._log_usage(
                UsageRecord(
                    operation="embed",
                    role="embed",
                    provider=active_provider.name,
                    model=resolved_model,
                    latency_ms=round(elapsed, 2),
                    status="error",
                    error=str(exc),
                )
            )
            raise

        self._write_cache(
            key,
            {
                "embeddings": result.embeddings,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
            },
        )
        self._log_usage(
            UsageRecord(
                operation="embed",
                role="embed",
                provider=result.provider,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                total_tokens=result.prompt_tokens,
                latency_ms=result.latency_ms,
                cached=False,
                status="ok",
            )
        )
        return result.embeddings


class FakeLLM(LLMClient):
    """Test-friendly LLMClient backed by scripted FakeProvider responses."""

    def __init__(
        self,
        responses: Iterable[str | Exception | Callable[[list[LLMMessage]], str]] | None = None,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_dim: int = 64,
        config: ForgeConfig | None = None,
        cache_dir: Path | str | None = None,
        usage_log_path: Path | str | None = None,
    ) -> None:
        self.fake_provider = FakeProvider(
            responses=responses,
            embed_fn=embed_fn,
            embed_dim=embed_dim,
        )
        super().__init__(
            provider=self.fake_provider,
            config=config,
            cache_dir=cache_dir,
            usage_log_path=usage_log_path,
            cache_enabled=cache_dir is not None,
        )

    def queue_response(
        self, response: str | Exception | Callable[[list[LLMMessage]], str]
    ) -> None:
        """Append a scripted response to the underlying FakeProvider."""
        self.fake_provider.queue_response(response)
