"""Tests for Gemini provider profile, rate limits, per-role routing, privacy, and live smoke."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import AgentDefinition, AgentOutput, AnswerRecord, Citation
from forge.cli import app
from forge.config import ForgeConfig, load_config
from forge.evals import EvalCase, EvalProvenance, EvalSet
from forge.ingest import ingest_folder
from forge.judge import Judge
from forge.llm import (
    GeminiProvider,
    LLMClient,
    LLMMessage,
    LLMProviderError,
    OllamaProvider,
)
from forge.report import build_report
from forge.retrieval import HybridIndex

ROOT = Path(__file__).resolve().parent.parent
REHEARSAL_UNSUPPORTED = ROOT / "fixtures" / "rehearsal" / "unsupported_scope"


class _SmokeSchema(BaseModel):
    status: str
    count: int


def test_profile_selection_and_toml_strips_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("forge.config._load_dotenv", lambda _p: None)
    monkeypatch.delenv("FORGE_PROFILE", raising=False)
    monkeypatch.delenv("FORGE_GEMINI_API_KEY", raising=False)

    default_cfg = load_config()
    assert default_cfg.profile == "default"
    assert default_cfg.endpoints.provider == "ollama"

    gemini_cfg = load_config(profile="gemini")
    assert gemini_cfg.profile == "gemini"
    assert gemini_cfg.endpoints.provider == "gemini"
    assert (
        gemini_cfg.endpoints.gemini_base_url
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert gemini_cfg.models.agent == "gemini-3.5-flash-lite"
    assert gemini_cfg.models.judge == "gemini-4-hydrogen"
    assert gemini_cfg.models.embed == "gemini-embedding-001"
    assert "gemini_api_key" not in gemini_cfg.model_dump()["endpoints"]
    assert gemini_cfg.privacy_notice() is not None

    monkeypatch.setenv("FORGE_PROFILE", "gemini")
    env_cfg = load_config()
    assert env_cfg.profile == "gemini"
    assert env_cfg.endpoints.provider == "gemini"

    # Even if an adversarial config has gemini_api_key in TOML, load_config ignores it
    bad_toml = tmp_path / "forge.gemini.toml"
    bad_toml.write_text(
        '[endpoints]\nprovider = "gemini"\ngemini_api_key = "SHOULD_NEVER_LOAD"\n',
        encoding="utf-8",
    )
    loaded = load_config(config_path=bad_toml, profile="gemini")
    assert "SHOULD_NEVER_LOAD" not in loaded.model_dump_json()

    with pytest.raises(FileNotFoundError):
        load_config(profile="does_not_exist")


def test_missing_and_invalid_key_errors_never_leak_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("forge.config._load_dotenv", lambda _p: None)
    monkeypatch.delenv("FORGE_GEMINI_API_KEY", raising=False)
    cfg = load_config(profile="gemini")
    cfg.cache.enabled = False
    client = LLMClient.from_config(cfg)
    with pytest.raises(LLMProviderError) as missing_exc:
        client.chat("hello", role="agent")
    assert "FORGE_GEMINI_API_KEY" in str(missing_exc.value)

    secret = "AIzaSyFakeSecretValueForTesting123456"

    def _unauth_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == f"Bearer {secret}"
        return httpx.Response(
            401,
            json={"error": {"message": f"Invalid key {secret}"}},
        )

    prov = GeminiProvider(
        api_key=secret,
        max_http_retries=2,
        backoff_seconds=0.01,
        transport=httpx.MockTransport(_unauth_handler),
    )
    usage_log = tmp_path / "usage.jsonl"
    err_client = LLMClient(prov, config=cfg, usage_log_path=usage_log, cache_enabled=False)
    with pytest.raises(LLMProviderError) as auth_exc:
        err_client.chat("hello", role="agent")
    assert "FORGE_GEMINI_API_KEY" in str(auth_exc.value)
    assert secret not in str(auth_exc.value)
    assert secret not in usage_log.read_text(encoding="utf-8")


def test_gemini_429_backoff_quota_exhaustion_and_call_budget(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def _flaky_429(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.01"},
                json={"error": {"message": "RESOURCE_EXHAUSTED"}},
            )
        return httpx.Response(
            200,
            json={
                "model": "gemini-3.5-flash-lite",
                "choices": [{"message": {"content": '{"status":"ok","count":3}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    prov = GeminiProvider(
        api_key="test-key",
        requests_per_minute=None,
        max_concurrency=1,
        max_http_retries=3,
        backoff_seconds=0.01,
        transport=httpx.MockTransport(_flaky_429),
    )
    assert prov.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

    cfg = ForgeConfig(budgets={"max_calls_per_run": 1})  # type: ignore[arg-type]
    client = LLMClient(prov, config=cfg, cache_dir=tmp_path / "cache")
    parsed = client.json("Return status", schema=_SmokeSchema, model="gemini-3.5-flash-lite")
    assert parsed.status == "ok"
    assert attempts["count"] == 3

    # Cached replay succeeds even when call budget == 1
    replayed = client.json("Return status", schema=_SmokeSchema, model="gemini-3.5-flash-lite")
    assert replayed.count == 3

    # Second uncached live call hits per-run call budget cleanly
    with pytest.raises(LLMProviderError) as budget_exc:
        client.chat("Different prompt", model="gemini-3.5-flash-lite")
    assert "call budget exhausted" in str(budget_exc.value).lower()

    # Persistent 429 raises actionable quota-exhausted error
    always_429 = GeminiProvider(
        api_key="test-key",
        requests_per_minute=None,
        max_http_retries=1,
        backoff_seconds=0.01,
        transport=httpx.MockTransport(lambda req: httpx.Response(429, text="quota exceeded")),
    )
    with pytest.raises(LLMProviderError) as quota_exc:
        always_429.chat([LLMMessage(role="user", content="hi")], model="gemini-4-hydrogen",
                        temperature=0.0, seed=42, json_mode=False)
    assert "429" in str(quota_exc.value)
    assert "quota exhausted" in str(quota_exc.value).lower()


def test_cache_key_separation_between_ollama_and_gemini(tmp_path: Path) -> None:
    calls = {"ollama": 0, "gemini": 0, "ollama_emb": 0, "gemini_emb": 0}

    def _ollama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/embed"):
            calls["ollama_emb"] += 1
            return httpx.Response(200, json={"model": "m", "embeddings": [[0.1, 0.2]]})
        calls["ollama"] += 1
        return httpx.Response(
            200,
            json={"model": "m", "message": {"role": "assistant", "content": "from ollama"}},
        )

    def _gemini_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            calls["gemini_emb"] += 1
            return httpx.Response(
                200, json={"model": "m", "data": [{"index": 0, "embedding": [0.9, 0.8]}]}
            )
        calls["gemini"] += 1
        return httpx.Response(
            200,
            json={"model": "m", "choices": [{"message": {"content": "from gemini"}}]},
        )

    cache_dir = tmp_path / "shared_cache"
    ollama_client = LLMClient(
        OllamaProvider(transport=httpx.MockTransport(_ollama_handler)),
        cache_dir=cache_dir,
    )
    gemini_client = LLMClient(
        GeminiProvider(
            api_key="test-key",
            requests_per_minute=None,
            transport=httpx.MockTransport(_gemini_handler),
        ),
        cache_dir=cache_dir,
    )

    assert ollama_client.chat("same prompt", model="shared-model").content == "from ollama"
    assert gemini_client.chat("same prompt", model="shared-model").content == "from gemini"
    assert gemini_client.chat("same prompt", model="shared-model").cached is True
    assert calls["ollama"] == 1
    assert calls["gemini"] == 1

    assert ollama_client.embed(["hello"], model="shared-emb") == [[0.1, 0.2]]
    assert gemini_client.embed(["hello"], model="shared-emb") == [[0.9, 0.8]]
    assert calls["ollama_emb"] == 1
    assert calls["gemini_emb"] == 1


def test_per_role_providers_bm25_fallback_and_family_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_GEMINI_API_KEY", "test-secret-not-for-logs")
    monkeypatch.setenv("FORGE_JUDGE_PROVIDER", "gemini")
    monkeypatch.setenv("FORGE_EVAL_WRITER_PROVIDER", "gemini")
    monkeypatch.setenv("FORGE_AGENT_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "gemini-4-hydrogen")
    monkeypatch.setenv("FORGE_EMBED_MODEL", "")

    cfg = load_config()
    assert cfg.provider_for_role("agent") == "ollama"
    assert cfg.provider_for_role("judge") == "gemini"
    assert cfg.provider_for_role("eval_writer") == "gemini"
    assert cfg.provider_for_role("embed") == "ollama"
    assert cfg.hosted_providers_used() == ["gemini"]

    client = LLMClient.from_config(cfg, cache_dir=tmp_path / "cache")
    assert client._provider_for_role("agent").name == "ollama"
    assert client._provider_for_role("judge").name == "gemini"

    # Empty embed model skips embedding endpoint cleanly and HybridIndex falls back to BM25
    with pytest.raises(LLMProviderError):
        client.embed(["hello"])
    _, chunks, _ = ingest_folder(REHEARSAL_UNSUPPORTED / "docs", chunk_size=300, overlap=20)
    bm25_idx = HybridIndex.build(chunks, llm_client=None)
    assert bm25_idx.embeddings is None

    # Same-family warning fires when both agent and judge are Gemini variants
    gemini_cfg = load_config(profile="gemini")
    gemini_client = LLMClient(
        GeminiProvider(
            api_key="test-secret-not-for-logs",
            requests_per_minute=None,
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "model": "gemini-4-hydrogen",
                        "choices": [{"message": {"content": '{"claims":[]}'}}],
                    },
                )
            ),
        ),
        config=gemini_cfg,
        cache_enabled=False,
    )
    defn = AgentDefinition.load_yaml(ROOT / "fixtures" / "hand_agent.yaml")
    defn.model.provider = "gemini"
    defn.model.name = "gemini-3.5-flash-lite"
    defn.meta.provenance.profile = "gemini"
    defn.meta.provenance.providers = gemini_cfg.providers_by_role()
    defn.meta.provenance.model_names = {
        "agent": "gemini-3.5-flash-lite",
        "judge": "gemini-4-hydrogen",
        "eval_writer": "gemini-4-hydrogen",
        "embed": "gemini-embedding-001",
    }
    prov = EvalProvenance(prompt_file="p", prompt_hash="h", model="gemini-4-hydrogen", seed=42)
    case = EvalCase(
        id="DEV-001",
        category="out_of_scope",
        question="What is unmentioned pricing?",
        expected_behavior="refuse",
        provenance=prov,
    )
    rec = AnswerRecord(
        question=case.question,
        output=AgentOutput(answer="The documents don't cover that.", refused=True),
    )
    scorecard = Judge(gemini_client, config=gemini_cfg).evaluate_suite(
        defn, [case], [rec], {c.chunk_id: c for c in chunks}, split="held_out"
    )
    assert scorecard.same_model_as_agent is False
    assert scorecard.same_family_as_agent is True

    run_dir = tmp_path / "run_meta"
    run_dir.mkdir()
    defn.save_yaml(run_dir / "definition.yaml")
    scorecard.save(run_dir / "final_held_out_eval")
    EvalSet(
        agent_name=defn.meta.name,
        eval_writer_model="gemini-4-hydrogen",
        agent_model="gemini-3.5-flash-lite",
        same_model_as_agent=False,
        same_family_as_agent=True,
        seed=42,
        held_out_cases=[case],
    ).save(run_dir)
    rep_path, _ = build_report(
        run_dir,
        pipeline_state={
            "profile": "gemini",
            "providers": gemini_cfg.providers_by_role(),
            "models": defn.meta.provenance.model_names,
            "completed_stages": ["report"],
        },
    )
    rep_txt = rep_path.read_text(encoding="utf-8")
    assert "**Profile**: `gemini`" in rep_txt
    assert '"agent": "gemini"' in rep_txt
    assert "Same-Model-Family Warning" in rep_txt
    assert "Same-Model Warning" not in rep_txt

    # When exact same model is used for agent and judge, BOTH warnings fire
    same_defn = defn.model_copy(deep=True)
    same_defn.model.name = "gemini-4-hydrogen"
    same_sc = Judge(gemini_client, config=gemini_cfg).evaluate_suite(
        same_defn, [case], [rec], {c.chunk_id: c for c in chunks}, split="held_out"
    )
    assert same_sc.same_model_as_agent is True
    assert same_sc.same_family_as_agent is True
    same_dir = tmp_path / "run_same_model"
    same_dir.mkdir()
    same_defn.save_yaml(same_dir / "definition.yaml")
    same_sc.save(same_dir / "final_held_out_eval")
    same_rep_path, _ = build_report(
        same_dir,
        pipeline_state={
            "profile": "gemini",
            "providers": gemini_cfg.providers_by_role(),
            "completed_stages": ["report"],
        },
    )
    same_rep_txt = same_rep_path.read_text(encoding="utf-8")
    assert "Same-Model Warning" in same_rep_txt
    assert "Same-Model-Family Warning" in same_rep_txt


def test_cli_run_profile_gemini_fast_and_secret_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "AIzaSyCliMockSecretKey9876543210"
    monkeypatch.setenv("FORGE_GEMINI_API_KEY", secret)

    _, chunks, _ = ingest_folder(REHEARSAL_UNSUPPORTED / "docs", chunk_size=300, overlap=20)
    first_chunk = chunks[0]

    def _gemini_cli_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content.decode("utf-8"))
            data = [
                {"index": i, "embedding": [0.5, 0.5, 0.5]} for i, _ in enumerate(body["input"])
            ]
            return httpx.Response(200, json={"model": body["model"], "data": data})

        body = json.loads(request.content.decode("utf-8"))
        prompt = body["messages"][-1]["content"]
        if "Generate ONLY genuinely necessary clarifying questions" in prompt:
            content = json.dumps({"questions": []})
        elif "synthesizing a structured requirements specification" in prompt:
            content = json.dumps(
                {
                    "scope": "Office printer setup only.",
                    "audience": "Staff",
                    "tone": "Concise",
                    "must_answer_topics": [
                        {
                            "id": "REQ-001",
                            "title": "Printer Setup",
                            "description": "CORP-PRINT network",
                            "priority": "high",
                        }
                    ],
                    "must_refuse_topics": [],
                    "assumptions": [],
                    "capabilities_needed": [],
                }
            )
        elif "Decompose the brief" in prompt:
            content = json.dumps(
                {
                    "topics": [
                        {
                            "topic_id": "TOP-001",
                            "title": "Printer Setup",
                            "description": "CORP-PRINT wireless network",
                            "source_req_id": "REQ-001",
                        }
                    ]
                }
            )
        elif "You are the Groundability Guard." in prompt:
            line = first_chunk.text.splitlines()[0]
            content = json.dumps(
                {
                    "status": "SUPPORTED",
                    "quotes": [{"chunk_id": first_chunk.chunk_id, "quote": line}],
                    "reasoning": "Matched chunk text.",
                    "gap_note": None,
                }
            )
        elif "Propose which tools" in prompt:
            content = json.dumps({"proposed_tools": []})
        elif 'Return ONLY a JSON object:\n{"role":' in prompt:
            content = json.dumps({"role": "Printer setup assistant."})
        elif 'Return ONLY a JSON object:\n{"scope":' in prompt:
            content = json.dumps({"scope": "Printer setup only."})
        elif 'Return ONLY a JSON object:\n{"audience_tone":' in prompt:
            content = json.dumps({"audience_tone": "Concise."})
        elif 'Return ONLY a JSON object:\n{"grounding_rules":' in prompt:
            content = json.dumps({"grounding_rules": "Cite quoted text."})
        elif 'Return ONLY a JSON object:\n{"refusal_policy":' in prompt:
            content = json.dumps(
                {"refusal_policy": "Say documents don't cover unsupported topics."}
            )
        elif 'Return ONLY a JSON object:\n{"format_rules":' in prompt:
            content = json.dumps(
                {"format_rules": "Return output JSON with doc_id, chunk_id, quote."}
            )
        elif 'Return ONLY a JSON object:\n{"extra_rules":' in prompt:
            content = json.dumps({"extra_rules": []})
        elif "Chunk Reference:" in prompt:
            content = json.dumps(
                {
                    "cases": [
                        {
                            "question": "Which wireless network is used?",
                            "key_facts": [first_chunk.text.splitlines()[0][:35]],
                            "notes": "ok",
                        }
                    ]
                }
            )
        elif "out-of-scope evaluation questions" in prompt:
            content = json.dumps(
                {
                    "cases": [
                        {"question": f"What is OOS topic {i}?", "subtype": "oos"}
                        for i in range(4)
                    ]
                }
            )
        elif "Determine whether the question below can be answered using ONLY" in prompt:
            content = json.dumps({"solvable": True, "reason": "ok"})
        elif "candidate out-of-scope question IS actually answerable" in prompt:
            content = json.dumps({"answerable_from_docs": False, "reason": "oos"})
        elif "Extract all atomic factual claims" in prompt:
            content = json.dumps({"claims": [first_chunk.text.splitlines()[0][:35]]})
        elif "Verify whether EACH numbered claim" in prompt:
            content = json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_index": 1,
                            "supported": True,
                            "quote": first_chunk.text.splitlines()[0][:35],
                        }
                    ]
                }
            )
        elif "Evaluate correctness" in prompt:
            content = json.dumps({"correct": True, "reason": "ok"})
        elif "Propose at most 2 targeted patches" in prompt:
            content = json.dumps({"patches": []})
        elif "Authoritative Output Contract Schema" in prompt:
            content = json.dumps(
                {
                    "action": "final",
                    "answer": f"Use {first_chunk.text.splitlines()[0][:35]}",
                    "citations": [
                        Citation(
                            doc_id=first_chunk.doc_id,
                            chunk_id=first_chunk.chunk_id,
                            quote=first_chunk.text.splitlines()[0][:35],
                        ).model_dump()
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.9,
                }
            )
        else:
            content = json.dumps({"cases": [], "diagnosis_items": [], "overall_summary": ""})
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    transport = httpx.MockTransport(_gemini_cli_handler)
    orig_from_config = forge_cli.LLMClient.from_config

    def _patched_from_config(cls, config=None, **kwargs):
        kwargs["cache_dir"] = tmp_path / "cache"
        c = orig_from_config(config, **kwargs)
        c.provider = GeminiProvider(
            api_key=secret,
            requests_per_minute=None,
            transport=transport,
        )
        for rname in list(c.providers_by_role):
            c.providers_by_role[rname] = c.provider
        return c

    monkeypatch.setattr(forge_cli.LLMClient, "from_config", classmethod(_patched_from_config))

    out_dir = tmp_path / "gemini_run"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--profile",
            "gemini",
            "--brief",
            str(REHEARSAL_UNSUPPORTED / "brief.txt"),
            "--docs",
            str(REHEARSAL_UNSUPPORTED / "docs"),
            "--out",
            str(out_dir),
            "--auto",
            "--fast",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Privacy notice: Hosted provider (gemini)" in result.output
    report_txt = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "**Profile**: `gemini`" in report_txt

    # Key must NEVER appear in any artifact or log inside out_dir
    for artifact in out_dir.rglob("*"):
        if artifact.is_file():
            txt = artifact.read_text(encoding="utf-8", errors="ignore")
            assert secret not in txt, f"API key leaked into {artifact}"


def test_no_key_shaped_strings_in_tracked_repo_files() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    pattern = re.compile(r"AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_-]{15,}|sk-[0-9A-Za-z]{20,}")
    hits: list[str] = []
    for rel in tracked:
        path = ROOT / rel
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(content):
            hits.append(rel)
    assert not hits, f"Key-shaped string found in tracked files: {hits}"


@pytest.mark.live
def test_live_ollama_json_and_embed() -> None:
    cfg = load_config(profile="default")
    try:
        resp = httpx.get(f"{cfg.endpoints.ollama_base_url.rstrip('/')}/api/tags", timeout=3.0)
        if resp.status_code != 200:
            pytest.skip("Ollama endpoint returned non-200 status.")
    except httpx.HTTPError:
        pytest.skip("Ollama server not reachable at configured base URL.")

    client = LLMClient.from_config(cfg)
    parsed = client.json(
        'Return ONLY JSON matching {"status":"ok","count":1}.',
        schema=_SmokeSchema,
        role="agent",
    )
    assert parsed.status
    vecs = client.embed(["ForgeAI live Ollama embedding smoke test."])
    assert len(vecs) == 1 and len(vecs[0]) > 10


@pytest.mark.live
def test_live_gemini_json_and_embed() -> None:
    cfg = load_config(profile="gemini")
    import os

    if not os.environ.get("FORGE_GEMINI_API_KEY", "").strip():
        pytest.skip("FORGE_GEMINI_API_KEY not set in environment or .env.")

    client = LLMClient.from_config(cfg)
    parsed = client.json(
        'Return ONLY JSON matching {"status":"ok","count":1}.',
        schema=_SmokeSchema,
        role="agent",
    )
    assert parsed.status
    vecs = client.embed(["ForgeAI live Gemini embedding smoke test."])
    assert len(vecs) == 1 and len(vecs[0]) > 10
