"""Hardening, robustness, chaos, time-budget, and rehearsal tests."""

from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter
from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import AgentDefinition
from forge.cli import app
from forge.evals import EvalGenerator
from forge.ingest import ingest_folder, load_documents
from forge.intake import IntakeInterviewer, RequirementsDocument
from forge.llm import FakeLLM, LLMClient, LLMMessage, LLMProviderError, OllamaProvider
from forge.pipeline import ForgePipeline
from forge.retrieval import HybridIndex

REHEARSAL_IT = Path(__file__).resolve().parent.parent / "fixtures" / "rehearsal" / "it_helpdesk"
REHEARSAL_UNSUPPORTED = (
    Path(__file__).resolve().parent.parent / "fixtures" / "rehearsal" / "unsupported_scope"
)


def test_ingest_robustness_large_non_utf8_odd_csv_and_scanned_pdf(tmp_path: Path) -> None:
    docs = tmp_path / "messy_docs"
    docs.mkdir()

    # 1. Empty text file
    (docs / "empty.txt").write_text("   \n", encoding="utf-8")

    # 2. Very large text file (exceeds custom cap -> truncated + warned)
    (docs / "huge.txt").write_text("alpha " * 500, encoding="utf-8")

    # 3. Non-UTF-8 Latin-1 file
    (docs / "latin1.txt").write_bytes("Café résumé naïve 42 units.".encode("latin-1"))

    # 4. UTF-16 BOM file
    (docs / "utf16.txt").write_bytes("System port 8443 active.".encode("utf-16"))

    # 5. Semicolon-delimited CSV with CP1252 bytes
    (docs / "semicolon.csv").write_bytes(
        "item;cost_eur;notes\nRouter;120;Café branch\nSwitch;250;Rack B\n".encode("cp1252")
    )

    # 6. Scanned/blank PDF
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (docs / "scan.pdf").open("wb") as fh:
        writer.write(fh)

    # 7. Unknown binary file
    (docs / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    res = load_documents(docs, max_file_bytes=400, max_doc_chars=200)
    by_id = {d.doc_id: d for d in res.documents}

    assert "latin1.txt" in by_id
    assert "utf16.txt" in by_id
    assert "semicolon.csv" in by_id
    assert by_id["semicolon.csv"].metadata["csv_profile"]["columns"] == [
        "item",
        "cost_eur",
        "notes",
    ]
    assert by_id["huge.txt"].metadata["truncated"] is True
    assert len(by_id["huge.txt"].text) <= 200

    assert any("Empty text file" in w for w in res.warnings)
    assert any("Large text file 'huge.txt'" in w for w in res.warnings)
    assert any("Scanned or empty PDF" in w for w in res.warnings)
    assert any("Unsupported file type '.png'" in w for w in res.warnings)


def test_ollama_provider_backoff_and_actionable_connect_error() -> None:
    # 1. ConnectError yields clear actionable guidance
    def connect_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    prov_down = OllamaProvider(transport=httpx.MockTransport(connect_fail))
    with pytest.raises(LLMProviderError) as exc_info:
        LLMClient(provider=prov_down, cache_enabled=False).chat("hi", model="qwen2.5:3b-instruct")
    assert "ollama serve" in str(exc_info.value)
    assert "ollama pull qwen2.5:3b-instruct" in str(exc_info.value)

    # 2. Transient TimeoutException retries and succeeds
    attempts = {"n": 0}

    def flaky_timeout(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b-instruct",
                "message": {"role": "assistant", "content": "ok"},
                "prompt_eval_count": 2,
                "eval_count": 1,
            },
        )

    prov_flaky = OllamaProvider(
        backoff_seconds=0.01,
        transport=httpx.MockTransport(flaky_timeout),
    )
    resp = LLMClient(provider=prov_flaky, cache_enabled=False).chat("hi")
    assert resp.content == "ok"
    assert attempts["n"] == 2


def test_short_corpus_scales_eval_quotas_and_vague_brief_survives(tmp_path: Path) -> None:
    _, chunks, profile = ingest_folder(REHEARSAL_UNSUPPORTED / "docs", chunk_size=300, overlap=20)
    index = HybridIndex.build(chunks, llm_client=None)

    # Chaos FakeLLM for intake with vague 1-sentence brief -> deterministic fallback works
    vague_llm = FakeLLM(responses=["", "{not json", "nope"] * 10)
    req = IntakeInterviewer(vague_llm).run("Help users with stuff.", profile, mode="auto")
    assert req.must_answer_topics[0].id == "REQ-001"

    # Short corpus eval generation scales quotas down with explicit note
    defn = AgentDefinition.load_yaml(
        Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"
    )
    defn.tools = []

    def _short_eval_responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Determine whether the question below can be answered using ONLY" in prompt:
            return json.dumps({"solvable": True, "reason": "ok"})
        if "Determine whether the candidate out-of-scope question IS actually answerable" in prompt:
            return json.dumps({"answerable_from_docs": False, "reason": "oos"})
        if "Chunk Reference:" in prompt:
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": "Which wireless network does the printer use?",
                            "key_facts": ["Connect to wireless network CORP-PRINT"],
                            "notes": "printer",
                        }
                    ]
                }
            )
        if "out-of-scope evaluation questions" in prompt:
            return json.dumps(
                {
                    "cases": [
                        {"question": f"OOS satellite pricing #{i}?", "subtype": "must_refuse"}
                        for i in range(1, 5)
                    ]
                }
            )
        return json.dumps({"cases": []})

    eval_set = EvalGenerator(FakeLLM(responses=[_short_eval_responder] * 30)).generate(
        defn, req, profile, index, out_dir=tmp_path
    )
    assert eval_set.scaled_quotas_note is not None
    assert "Short corpus" in eval_set.scaled_quotas_note
    assert "Quota Scaling Note" in (tmp_path / "eval_set_stats.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "chaos_payload",
    [
        "",
        "   ",
        "{malformed json !!!",
        '{"wrong_schema": 123, "refused": true}',
    ],
)
def test_chaos_bad_llm_outputs_degrade_without_crashing(
    tmp_path: Path, chaos_payload: str
) -> None:
    """Pipeline with broken/empty/wrong-schema LLM outputs degrades cleanly + writes report."""
    llm = FakeLLM(responses=[chaos_payload] * 400)
    out_dir = tmp_path / "chaos_run"
    state = ForgePipeline(llm, bm25_only=True).run(
        brief="Answer questions about office printer setup.",
        docs_dir=REHEARSAL_UNSUPPORTED / "docs",
        out_dir=out_dir,
        mode="auto",
        fast=True,
        max_iters=1,
    )
    assert "report" in state.completed_stages
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "definition.yaml").is_file()


def _rehearsal_responder(chunks_by_id: dict[str, str], *, unsupported_case: bool = False):
    cids = list(chunks_by_id.keys())
    counters = {"s": 0}

    def _respond(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Generate ONLY genuinely necessary clarifying questions" in prompt:
            return json.dumps({"questions": [{"id": "Q1", "question": "Audience?", "why": "tone"}]})
        if "simulating a pragmatic requester" in prompt:
            return json.dumps({"answers": [{"question_id": "Q1", "answer": "Internal staff."}]})
        if "synthesizing a structured requirements specification" in prompt:
            if unsupported_case:
                return json.dumps(
                    {
                        "scope": "Printer setup, satellite pricing, and legal indemnification.",
                        "audience": "Staff",
                        "tone": "Direct",
                        "must_answer_topics": [
                            {
                                "id": "REQ-001",
                                "title": "Office Printer Setup",
                                "description": "Wireless network CORP-PRINT and duplex PIN",
                                "priority": "high",
                            },
                            {
                                "id": "REQ-002",
                                "title": "Satellite Imagery Pricing",
                                "description": "Commercial satellite licensing rates",
                                "priority": "high",
                            },
                        ],
                        "must_refuse_topics": [],
                        "assumptions": [],
                        "capabilities_needed": [],
                    }
                )
            return json.dumps(
                {
                    "scope": "IT helpdesk VPN, MFA, onboarding, and password reset SLA.",
                    "audience": "Employees",
                    "tone": "Helpful",
                    "must_answer_topics": [
                        {
                            "id": "REQ-001",
                            "title": "VPN and MFA",
                            "description": "WireGuard port 51820 and FIDO2 MFA",
                            "priority": "high",
                        }
                    ],
                    "must_refuse_topics": [],
                    "assumptions": [],
                    "capabilities_needed": [],
                }
            )
        if "Decompose the brief, requirements, and clarifying Q&A" in prompt:
            if unsupported_case:
                return json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": "TOP-001",
                                "title": "Office Printer Setup",
                                "description": "Wireless network CORP-PRINT and duplex PIN",
                                "source_req_id": "REQ-001",
                            },
                            {
                                "topic_id": "TOP-002",
                                "title": "Satellite Imagery Pricing",
                                "description": "Commercial satellite licensing rates",
                                "source_req_id": "REQ-002",
                            },
                        ]
                    }
                )
            return json.dumps(
                {
                    "topics": [
                        {
                            "topic_id": "TOP-001",
                            "title": "VPN and MFA",
                            "description": "WireGuard port 51820 and FIDO2 MFA",
                            "source_req_id": "REQ-001",
                        }
                    ]
                }
            )
        if "You are the Groundability Guard." in prompt:
            if "Satellite Imagery" in prompt:
                return json.dumps(
                    {
                        "status": "UNSUPPORTED",
                        "quotes": [],
                        "reasoning": "Not in printer docs.",
                        "gap_note": "No satellite pricing in corpus.",
                    }
                )
            cid = next(
                (c for c in cids if "CORP-PRINT" in chunks_by_id[c] or "51820" in chunks_by_id[c]),
                cids[0],
            )
            quote = next(
                (
                    ln.strip()
                    for ln in chunks_by_id[cid].splitlines()
                    if "CORP-PRINT" in ln or "51820" in ln
                ),
                chunks_by_id[cid][:60],
            )
            return json.dumps(
                {
                    "status": "SUPPORTED",
                    "quotes": [{"chunk_id": cid, "quote": quote}],
                    "reasoning": "Verified in docs.",
                    "gap_note": None,
                }
            )
        if "Propose at most 2 targeted patches" in prompt:
            return json.dumps({"patches": []})
        if "Propose which tools from the registry" in prompt:
            return json.dumps({"proposed_tools": []})
        if 'Return ONLY a JSON object:\n{"role":' in prompt:
            return json.dumps({"role": "Internal support assistant."})
        if 'Return ONLY a JSON object:\n{"scope":' in prompt:
            return json.dumps({"scope": "Grounded documentation answers only."})
        if 'Return ONLY a JSON object:\n{"audience_tone":' in prompt:
            return json.dumps({"audience_tone": "Clear and concise."})
        if 'Return ONLY a JSON object:\n{"grounding_rules":' in prompt:
            return json.dumps({"grounding_rules": "Cite verbatim quoted evidence."})
        if 'Return ONLY a JSON object:\n{"refusal_policy":' in prompt:
            return json.dumps(
                {"refusal_policy": "State honestly that the documents don't cover this."}
            )
        if 'Return ONLY a JSON object:\n{"format_rules":' in prompt:
            return json.dumps({"format_rules": "Return output contract JSON."})
        if 'Return ONLY a JSON object:\n{"extra_rules":' in prompt:
            return json.dumps({"extra_rules": ["Keep answers brief."]})
        if "Chunk Reference:" in prompt:
            m = re.search(r"chunk_id=([^\s]+)", prompt)
            cid = m.group(1) if m else cids[0]
            txt = chunks_by_id[cid]
            line = next((ln.strip() for ln in txt.splitlines() if len(ln.strip()) > 10), txt)
            counters["s"] += 1
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": f"Rehearsal question #{counters['s']} on {cid}?",
                            "key_facts": [line[:45]],
                            "notes": cid,
                        }
                    ]
                }
            )
        if "REQUIRE combining facts from BOTH chunks below" in prompt:
            m_a = re.search(r"Chunk A: doc_id=[^\s]+ \| chunk_id=([^\s]+)", prompt)
            m_b = re.search(r"Chunk B: doc_id=[^\s]+ \| chunk_id=([^\s]+)", prompt)
            ca, cb = (m_a.group(1) if m_a else cids[0]), (m_b.group(1) if m_b else cids[0])
            la = next(
                (ln.strip() for ln in chunks_by_id[ca].splitlines() if len(ln.strip()) > 8),
                chunks_by_id[ca],
            )
            lb = next(
                (ln.strip() for ln in chunks_by_id[cb].splitlines() if len(ln.strip()) > 8),
                chunks_by_id[cb],
            )
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": f"Combine {ca} and {cb}?",
                            "key_facts": [la[:40], lb[:40]],
                            "notes": "multi",
                        }
                    ]
                }
            )
        if "out-of-scope evaluation questions" in prompt:
            return json.dumps(
                {
                    "cases": [
                        {"question": f"OOS question #{i}?", "subtype": "must_refuse"}
                        for i in range(1, 5)
                    ]
                }
            )
        if "edge-case evaluation questions" in prompt:
            return json.dumps({"cases": []})
        if "Determine whether the question below can be answered using ONLY" in prompt:
            return json.dumps({"solvable": True, "reason": "ok"})
        if "Determine whether the candidate out-of-scope question IS actually answerable" in prompt:
            return json.dumps({"answerable_from_docs": False, "reason": "oos"})
        if "User question:" in prompt:
            if "oos" in prompt.lower() or "satellite" in prompt.lower():
                return json.dumps(
                    {
                        "action": "final",
                        "answer": "The documents don't cover that.",
                        "citations": [],
                        "refused": True,
                        "refusal_reason": "Out of scope.",
                        "confidence": 0.95,
                    }
                )
            m_ret = re.search(
                r"\[1\] \(doc_id=([^ |]+) \| chunk_id=([^\s|]+)[^\n]*\n(.+?)(?:\n\n|\Z)",
                prompt,
                re.DOTALL,
            )
            if m_ret:
                did, cid = m_ret.group(1), m_ret.group(2)
                lines = [ln.strip() for ln in m_ret.group(3).splitlines() if ln.strip()]
                line = lines[-1] if lines else m_ret.group(3).strip()
            else:
                did, cid, line = "doc", cids[0], chunks_by_id[cids[0]].splitlines()[0]
            return json.dumps(
                {
                    "action": "final",
                    "answer": f"Document states: {line}",
                    "citations": [{"doc_id": did, "chunk_id": cid, "quote": line[:80]}],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.9,
                }
            )
        if "Extract all atomic factual claims" in prompt:
            ans = prompt.split("Answer:\n", 1)[1].split("\n\nReturn ONLY", 1)[0].strip()
            return json.dumps({"claims": [ans]})
        if "Verify whether EACH numbered claim" in prompt:
            ev = prompt.split("Evidence (Retrieved Passages & Tool Outputs):\n", 1)[1].split(
                "\n\nReturn ONLY", 1
            )[0]
            ln = next(
                (x.strip() for x in ev.splitlines() if x.strip() and not x.strip().startswith("[")),
                "",
            )
            return json.dumps(
                {"verdicts": [{"claim_index": 1, "supported": bool(ln), "quote": ln[:80]}]}
            )
        return json.dumps({"correct": True, "reason": "ok"})

    return _respond


def test_rehearsal_runs_and_time_budget_degradation(tmp_path: Path, monkeypatch) -> None:
    """Run both rehearsal sets unattended with --fast and verify --time-budget-min."""
    # Ensure onboarding_checklist.docx exists in rehearsal/it_helpdesk/docs
    docx_path = REHEARSAL_IT / "docs" / "onboarding_checklist.docx"
    assert docx_path.is_file()
    with zipfile.ZipFile(docx_path, "r") as zf:
        assert "word/document.xml" in zf.namelist()

    # 1. Rehearsal A: IT helpdesk (md + docx + txt) with --time-budget-min degradation
    _, chunks_a, _ = ingest_folder(REHEARSAL_IT / "docs", chunk_size=400, overlap=40)
    llm_a = FakeLLM(
        responses=[_rehearsal_responder({c.chunk_id: c.text for c in chunks_a})] * 400
    )
    out_a = tmp_path / "rehearsal_it"

    # Sleep briefly in 'generate' hook so tight 0.005 min (0.3s) time budget triggers degradation!
    def _slow_step() -> None:
        time.sleep(0.35)

    state_a = ForgePipeline(llm_a, bm25_only=True).run(
        brief=(REHEARSAL_IT / "brief.txt").read_text(encoding="utf-8"),
        docs_dir=REHEARSAL_IT / "docs",
        out_dir=out_a,
        mode="auto",
        fast=True,
        time_budget_min=0.005,
        stage_hooks={"generate": _slow_step},
    )
    assert state_a.time_budget_degraded is True
    assert state_a.time_budget_notes
    rep_a = (out_a / "report.md").read_text(encoding="utf-8")
    assert "Time Budget Degradation Notes" in rep_a
    assert AgentDefinition.load_yaml(out_a / "definition.yaml").attached_tool_names() == []

    # 2. Rehearsal B: Unsupported scope via CLI `forge run --auto --fast`
    _, chunks_b, _ = ingest_folder(REHEARSAL_UNSUPPORTED / "docs", chunk_size=400, overlap=40)
    llm_b = FakeLLM(
        responses=[
            _rehearsal_responder({c.chunk_id: c.text for c in chunks_b}, unsupported_case=True)
        ]
        * 400
    )
    monkeypatch.setattr(
        forge_cli.LLMClient,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: llm_b),
    )

    out_b = tmp_path / "rehearsal_unsupported"
    res = CliRunner().invoke(
        app,
        [
            "run",
            "--brief",
            str(REHEARSAL_UNSUPPORTED / "brief.txt"),
            "--docs",
            str(REHEARSAL_UNSUPPORTED / "docs"),
            "--out",
            str(out_b),
            "--auto",
            "--fast",
            "--bm25-only",
        ],
    )
    assert res.exit_code == 0
    assert (out_b / "report.md").is_file()
    req_b = RequirementsDocument.model_validate_json(
        (out_b / "requirements.json").read_text(encoding="utf-8")
    )
    assert req_b.groundability_report is not None
    assert req_b.groundability_report.unsupported_count >= 1
    assert any(r.from_gap for r in req_b.must_refuse_topics)
