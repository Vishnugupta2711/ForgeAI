"""Unit tests for forge.agent.schema (AgentDefinition YAML serialization & validation)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from forge.agent import (
    AgentDefinition,
    AgentMeta,
    AgentOutput,
    AgentRetrievalSettings,
    AttachedToolSpec,
    Citation,
    OutputContractConfig,
    SystemPromptSections,
)

HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def _minimal_definition() -> AgentDefinition:
    return AgentDefinition(
        meta=AgentMeta(name="test-agent", brief="Answer test questions."),
        system_prompt=SystemPromptSections(
            role="You are a careful assistant.",
            scope="Only answer from provided docs.",
            audience_tone="Concise and neutral.",
            grounding_rules="Cite doc_id, chunk_id, and verbatim quote.",
            refusal_policy="Refuse out-of-scope queries.",
            format_rules="Output strict JSON.",
            extra_rules=["Keep numbers exact."],
        ),
        tools=[
            AttachedToolSpec(
                name="calculator",
                justification_requirement_id="REQ-1",
            )
        ],
    )


def test_agent_definition_yaml_round_trip(tmp_path: Path) -> None:
    defn = _minimal_definition()
    yaml_path = tmp_path / "agent.yaml"
    defn.save_yaml(yaml_path)

    loaded = AgentDefinition.load_yaml(yaml_path)
    assert loaded.meta.name == "test-agent"
    assert loaded.attached_tool_names() == ["calculator"]
    rendered = loaded.system_prompt.render()
    assert "## Role" in rendered
    assert "## Additional Rules" in rendered
    assert "- Keep numbers exact." in rendered


def test_hand_agent_yaml_is_valid() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    assert defn.meta.name == "marine-sensor-assistant"
    assert defn.attached_tool_names() == ["csv_lookup", "calculator"]
    assert defn.model.max_tool_calls == 3


def test_invalid_retrieval_overlap_and_output_contract_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentRetrievalSettings(chunk_size=100, overlap=100)

    with pytest.raises(ValidationError):
        OutputContractConfig(
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            }
        )

    with pytest.raises(ValidationError):
        OutputContractConfig(
            json_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"doc_id": {"type": "string"}}},
                    },
                    "refused": {"type": "boolean"},
                    "refusal_reason": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                },
            }
        )


def test_agent_output_refusal_consistency() -> None:
    out = AgentOutput(
        answer="I cannot answer that.",
        citations=[Citation(doc_id="d1", chunk_id="c1", quote="q")],
        refused=True,
        confidence=0.9,
    )
    assert out.refusal_reason is not None
