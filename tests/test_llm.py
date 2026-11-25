"""Unit tests for forge.llm (LLMClient, providers, repair-retry, caching, usage logging)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, Field

from forge.config import EndpointsConfig, ForgeConfig
from forge.llm import (
    FakeLLM,
    LLMClient,
    LLMValidationError,
    OllamaProvider,
    OpenAICompatProvider,
)


class SampleSchema(BaseModel):
    topic: str
    score: int = Field(ge=1, le=10)


def test_fake_llm_json_fenced_and_prose() -> None:
    llm = FakeLLM(
        responses=[
            'Here is the output:\n```json\n{"topic": "retrieval", "score": 8}\n```\nDone!'
        ]
    )
    result = llm.json("Classify topic", schema=SampleSchema)
    assert result == SampleSchema(topic="retrieval", score=8)
    assert len(llm.fake_provider.chat_calls) == 1
    assert llm.fake_provider.chat_calls[0]["json_mode"] is True


def test_fake_llm_repair_retry_recovers_from_malformed_and_invalid_json() -> None:
    llm = FakeLLM(
        responses=[
            "{bad json",
            '{"topic": "retrieval", "score": 99}',  # fails ge=1, le=10 validation
            '{"topic": "retrieval", "score": 7}',
        ]
    )
    result = llm.json("Return topic and score", schema=SampleSchema, max_repairs=2)
    assert result == SampleSchema(topic="retrieval", score=7)
    assert len(llm.fake_provider.chat_calls) == 3

    # Verify repair prompt fed back the validation error to the model
    third_call_messages = llm.fake_provider.chat_calls[2]["messages"]
    assert any("Error details:" in m["content"] for m in third_call_messages)
    assert any("less than or equal to 10" in m["content"] for m in third_call_messages)


def test_fake_llm_repair_retry_exhaustion_raises() -> None:
    llm = FakeLLM(responses=["not json", "still not json", '{"topic": "x"}'])
    with pytest.raises(LLMValidationError) as exc_info:
        llm.json("Return topic and score", schema=SampleSchema, max_repairs=2)
    assert len(exc_info.value.errors) == 3
    assert len(llm.fake_provider.chat_calls) == 3


def test_disk_cache_hit_and_usage_logging(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    usage_log = tmp_path / "runs" / "usage.jsonl"

    llm1 = FakeLLM(
        responses=['{"topic": "cache", "score": 5}'],
        cache_dir=cache_dir,
        usage_log_path=usage_log,
    )
    first = llm1.chat("hello world", json_mode=True)
    assert first.cached is False
    assert len(llm1.fake_provider.chat_calls) == 1

    # Second client sharing same cache_dir should hit disk cache without invoking provider
    llm2 = FakeLLM(
        responses=[],
        cache_dir=cache_dir,
        usage_log_path=usage_log,
    )
    second = llm2.chat("hello world", json_mode=True)
    assert second.cached is True
    assert second.content == first.content
    assert len(llm2.fake_provider.chat_calls) == 0

    # Different temperature misses cache
    llm2.queue_response("fresh response")
    third = llm2.chat("hello world", temperature=0.7, json_mode=True)
    assert third.cached is False
    assert third.content == "fresh response"
    assert len(llm2.fake_provider.chat_calls) == 1

    # Embed cache hit test
    v1 = llm1.embed(["alpha", "beta"])
    v2 = llm2.embed(["alpha", "beta"])
    assert v1 == v2
    assert len(llm1.fake_provider.embed_calls) == 1
    assert len(llm2.fake_provider.embed_calls) == 0

    # Verify JSONL usage log written
    records = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert records[0]["cached"] is False
    assert records[1]["cached"] is True
    assert records[4]["operation"] == "embed"
    assert records[4]["cached"] is True


def test_ollama_provider_with_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode("utf-8"))
            assert body["format"] == "json"
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b-instruct",
                    "message": {"role": "assistant", "content": '{"topic":"ok","score":9}'},
                    "prompt_eval_count": 12,
                    "eval_count": 8,
                },
            )
        if request.url.path == "/api/embed":
            return httpx.Response(
                200,
                json={"model": "nomic-embed-text", "embeddings": [[0.1, 0.2, 0.3]]},
            )
        return httpx.Response(404)

    provider = OllamaProvider(transport=httpx.MockTransport(handler))
    client = LLMClient(provider=provider, cache_enabled=False)
    parsed = client.json("test", schema=SampleSchema)
    assert parsed == SampleSchema(topic="ok", score=9)
    vecs = client.embed(["hello"])
    assert vecs == [[0.1, 0.2, 0.3]]


def test_openai_compat_provider_and_from_config() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-key"
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "custom-model",
                    "choices": [{"message": {"content": "pong"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                },
            )
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": [0.5, 0.5]}],
                    "usage": {"prompt_tokens": 2},
                },
            )
        return httpx.Response(404)

    provider = OpenAICompatProvider(
        base_url="http://localhost:8000",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    client = LLMClient(provider=provider, cache_enabled=False)
    resp = client.chat("ping")
    assert resp.content == "pong"
    assert resp.total_tokens == 4
    assert client.embed("hi") == [[0.5, 0.5]]

    cfg = ForgeConfig(
        endpoints=EndpointsConfig(
            provider="openai_compat",
            openai_compat_base_url="http://localhost:8000",
        )
    )
    built = LLMClient.from_config(cfg)
    assert isinstance(built.provider, OpenAICompatProvider)
