"""Unit tests for forge.agent.runtime (tool loop, unattached tool rejection, repair)."""

from __future__ import annotations

import json
from pathlib import Path

from forge.agent import AgentDefinition, AgentRuntime
from forge.ingest import ingest_folder
from forge.llm import FakeLLM
from forge.retrieval import HybridIndex
from forge.tools import ToolContext

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"
HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def _build_fixture_index(llm: FakeLLM, chunk_size: int = 400, overlap: int = 40):
    ingest_res, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=chunk_size, overlap=overlap)
    index = HybridIndex.build(chunks, llm_client=llm)
    tool_ctx = ToolContext.from_documents(ingest_res.documents)
    return index, tool_ctx


def test_runtime_tool_loop_and_citations() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "csv_lookup",
                    "arguments": {
                        "table": "sensors.csv",
                        "filters": [{"column": "active", "op": "eq", "value": "true"}],
                    },
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": "60 + 90 + 75"},
                }
            ),
            json.dumps(
                {
                    "action": "final",
                    "answer": "Three active sensors have 225 combined battery days.",
                    "citations": [
                        {
                            "doc_id": "sensors.csv",
                            "chunk_id": "sensors.csv::row-1",
                            "quote": "sensor_id: SNS-101",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.92,
                }
            ),
        ]
    )
    index, tool_ctx = _build_fixture_index(llm)
    runtime = AgentRuntime(defn, index, llm_client=llm, tool_context=tool_ctx)

    record = runtime.answer("Total battery days of active sensors?")
    assert record.output.refused is False
    assert "225" in record.output.answer
    assert len(record.tool_calls) == 2
    assert [tc.tool_name for tc in record.tool_calls] == ["csv_lookup", "calculator"]
    assert all(tc.ok for tc in record.tool_calls)
    assert len(record.retrieved_chunks) > 0


def test_runtime_rejects_unattached_tool() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    # http_api is NOT attached in hand_agent.yaml
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "http_api",
                    "arguments": {"endpoint": "forecast", "latitude": 10.0, "longitude": 20.0},
                }
            ),
            json.dumps(
                {
                    "action": "final",
                    "answer": (
                        "I cannot fetch live weather forecasts because no weather tool "
                        "is attached."
                    ),
                    "citations": [],
                    "refused": True,
                    "refusal_reason": (
                        "External weather queries are outside scope and http_api "
                        "is not attached."
                    ),
                    "confidence": 0.95,
                }
            ),
        ]
    )
    index, tool_ctx = _build_fixture_index(llm)
    runtime = AgentRuntime(defn, index, llm_client=llm, tool_context=tool_ctx)

    record = runtime.answer("What is the weather forecast at 10N 20E?")
    assert record.output.refused is True
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_name == "http_api"
    assert record.tool_calls[0].ok is False
    assert "Rejected unattached tool 'http_api'" in (record.tool_calls[0].error or "")


def test_runtime_enforces_max_tool_calls() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.model.max_tool_calls = 2
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": "1 + 1"},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": "2 + 2"},
                }
            ),
            # 3rd tool call exceeds max_tool_calls=2 -> runtime sends force_final prompt
            json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": "3 + 3"},
                }
            ),
            json.dumps(
                {
                    "answer": "Computed intermediate results: 2 and 4.",
                    "citations": [],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.8,
                }
            ),
        ]
    )
    index, tool_ctx = _build_fixture_index(llm)
    runtime = AgentRuntime(defn, index, llm_client=llm, tool_context=tool_ctx)

    record = runtime.answer("Compute 1+1 and 2+2 and 3+3")
    assert len(record.tool_calls) == 2
    assert any("max_tool_calls (2) reached" in step for step in record.trace)
    assert record.output.answer == "Computed intermediate results: 2 and 4."


def test_runtime_contract_violation_repair_and_fallback() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    llm = FakeLLM(
        responses=[
            "{not valid json",
            json.dumps({"answer": "Missing required fields only", "confidence": 5.0}),
            json.dumps(
                {
                    "answer": "Deploy hydrophone pods at 45 meters below mean sea level.",
                    "citations": [
                        {
                            "doc_id": "acoustic_guide.md",
                            "chunk_id": "acoustic_guide.md::c1",
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.9,
                }
            ),
        ]
    )
    index, tool_ctx = _build_fixture_index(llm)
    runtime = AgentRuntime(defn, index, llm_client=llm, tool_context=tool_ctx)

    record = runtime.answer("What depth should hydrophone pods be deployed?")
    assert record.output.refused is False
    assert "45 meters" in record.output.answer
    assert any("output_contract validation error" in t for t in record.trace)

    # Exhausted repair retries return safe refusal fallback instead of crashing
    bad_llm = FakeLLM(responses=["bad1", "bad2", "bad3"])
    bad_runtime = AgentRuntime(defn, index, llm_client=bad_llm, tool_context=tool_ctx)
    fallback_record = bad_runtime.answer("Any question")
    assert fallback_record.output.refused is True
    assert fallback_record.output.confidence == 0.0
    assert fallback_record.contract_repairs == 2
    assert fallback_record.contract_failed is True
    assert len(bad_llm.fake_provider.chat_calls) == 3
    assert "failed output contract validation" in (fallback_record.output.refusal_reason or "")


def test_runtime_min_score_gate_and_query_rewrite() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.retrieval.query_rewrite = True
    defn.retrieval.rerank = True
    defn.retrieval.min_score = 0.99

    llm = FakeLLM(
        responses=[
            json.dumps({"rewritten_query": "hydrophone deployment depth"}),
            json.dumps(
                {
                    "answer": "No sufficiently relevant documents were found.",
                    "citations": [],
                    "refused": True,
                    "refusal_reason": "No chunks passed the retrieval threshold.",
                    "confidence": 0.85,
                }
            ),
        ]
    )
    index, tool_ctx = _build_fixture_index(llm)
    runtime = AgentRuntime(defn, index, llm_client=llm, tool_context=tool_ctx)

    record = runtime.answer("Where do we put the pods?")
    assert record.effective_query == "hydrophone deployment depth"
    assert record.retrieved_chunks == []
    assert record.output.refused is True
    assert any("min_score gate" in t for t in record.trace)


def test_runtime_appends_authoritative_schema_despite_wrong_format_rules() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.retrieval.query_rewrite = False
    defn.system_prompt.format_rules = (
        "Each citation object MUST have ONLY keys 'doc_id' and 'chunk_id'. Omit 'quote'."
    )

    llm = FakeLLM(
        responses=[
            # First answer follows drifted format_rules (missing quote) -> repair retry 1 fixes it
            json.dumps(
                {
                    "answer": "Deploy hydrophone pods at 45 meters.",
                    "citations": [
                        {"doc_id": "acoustic_guide.md", "chunk_id": "acoustic_guide.md::c1"}
                    ],
                    "refused": False,
                    "confidence": 0.85,
                }
            ),
            json.dumps(
                {
                    "answer": "Deploy hydrophone pods at 45 meters below mean sea level.",
                    "citations": [
                        {
                            "doc_id": "acoustic_guide.md",
                            "chunk_id": "acoustic_guide.md::c1",
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.91,
                }
            ),
        ]
    )
    index, tool_ctx = _build_fixture_index(llm)
    runtime = AgentRuntime(defn, index, llm_client=llm, tool_context=tool_ctx)
    record = runtime.answer("Where should hydrophone pods be deployed?")

    assert record.output.refused is False
    assert record.contract_failed is False
    assert record.contract_repairs == 1
    assert len(record.output.citations) == 1
    assert record.output.citations[0].quote

    # Initial user prompt MUST include machine-generated contract schema + quote key
    first_user_msg = llm.fake_provider.chat_calls[0]["messages"][-1]["content"]
    assert "Authoritative Output Contract Schema" in first_user_msg
    assert "chunk_id, doc_id, quote" in first_user_msg
    assert json.dumps(defn.output_contract.json_schema, indent=2) in first_user_msg

    # Repair prompt MUST include exact Pydantic validation error + JSON schema
    repair_user_msg = llm.fake_provider.chat_calls[1]["messages"][-1]["content"]
    assert "quote" in repair_user_msg
    assert "Field required" in repair_user_msg
    assert json.dumps(defn.output_contract.json_schema, indent=2) in repair_user_msg
