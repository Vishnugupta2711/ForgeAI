"""Declarative AgentDefinition schema, YAML serialization, and runtime answer models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Citation(BaseModel):
    """Source citation backing an agent answer."""

    doc_id: str
    chunk_id: str
    quote: str


class AgentOutput(BaseModel):
    """Structured answer contract enforced on every agent response."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_refusal_consistency(self) -> AgentOutput:
        if self.refused and not (self.refusal_reason and self.refusal_reason.strip()):
            self.refusal_reason = "Request is outside the grounded scope of the documents."
        return self


def default_output_contract_schema() -> dict[str, Any]:
    """Return the canonical JSON schema derived from AgentOutput."""
    return AgentOutput.model_json_schema()


class AgentProvenance(BaseModel):
    """Provenance metadata recording models, providers, profile, and prompt template hashes."""

    profile: str = "default"
    providers: dict[str, str] = Field(default_factory=dict)
    model_names: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    generator_version: str = "0.1.0"


class AgentMeta(BaseModel):
    """Identity and provenance metadata for an AgentDefinition."""

    name: str = Field(min_length=1)
    brief: str = Field(min_length=1)
    version: str = "1.0.0"
    provenance: AgentProvenance = Field(default_factory=AgentProvenance)


class SystemPromptSections(BaseModel):
    """Named prompt sections so optimizer patches can target individual sections."""

    role: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    audience_tone: str = Field(min_length=1)
    grounding_rules: str = Field(min_length=1)
    refusal_policy: str = Field(min_length=1)
    format_rules: str = Field(min_length=1)
    extra_rules: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Render the named sections into a single deterministic system prompt."""
        blocks = [
            f"## Role\n{self.role.strip()}",
            f"## Scope\n{self.scope.strip()}",
            f"## Audience & Tone\n{self.audience_tone.strip()}",
            f"## Grounding Rules\n{self.grounding_rules.strip()}",
            f"## Refusal Policy\n{self.refusal_policy.strip()}",
            f"## Format Rules\n{self.format_rules.strip()}",
        ]
        if self.extra_rules:
            extras = "\n".join(f"- {rule.strip()}" for rule in self.extra_rules if rule.strip())
            if extras:
                blocks.append(f"## Additional Rules\n{extras}")
        return "\n\n".join(blocks)


class AgentRetrievalSettings(BaseModel):
    """Declarative retrieval settings embedded in AgentDefinition."""

    chunk_size: int = Field(default=800, ge=64)
    overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank: bool = False
    query_rewrite: bool = False
    min_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_overlap(self) -> AgentRetrievalSettings:
        if self.overlap >= self.chunk_size:
            raise ValueError("retrieval.overlap must be smaller than retrieval.chunk_size.")
        return self


class AttachedToolSpec(BaseModel):
    """Tool attachment entry with justification back to requirements."""

    name: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
    justification_requirement_id: str = Field(min_length=1)


_REQUIRED_CONTRACT_FIELDS = {"answer", "citations", "refused", "refusal_reason", "confidence"}
_REQUIRED_CITATION_FIELDS = {"doc_id", "chunk_id", "quote"}


class OutputContractConfig(BaseModel):
    """Output contract JSON schema definition with mandatory grounding/refusal fields."""

    json_schema: dict[str, Any] = Field(default_factory=default_output_contract_schema)

    @field_validator("json_schema")
    @classmethod
    def _validate_contract_fields(cls, schema: dict[str, Any]) -> dict[str, Any]:
        props = schema.get("properties")
        if not isinstance(props, dict):
            raise ValueError("output_contract.json_schema must define a 'properties' object.")
        missing = _REQUIRED_CONTRACT_FIELDS - set(props.keys())
        if missing:
            raise ValueError(
                f"output_contract.json_schema missing required fields: {sorted(missing)}"
            )

        citations_schema = props.get("citations", {})
        items = citations_schema.get("items", {})
        if "$ref" in items:
            ref_name = str(items["$ref"]).rsplit("/", 1)[-1]
            items = schema.get("$defs", {}).get(ref_name, {})
        citation_props = items.get("properties", {}) if isinstance(items, dict) else {}
        missing_cit = _REQUIRED_CITATION_FIELDS - set(citation_props.keys())
        if missing_cit:
            raise ValueError(
                f"output_contract citations items missing fields: {sorted(missing_cit)}"
            )
        return schema


class AgentModelSettings(BaseModel):
    """Runtime model configuration for the generated agent."""

    provider: Literal["ollama", "openai_compat", "gemini"] = "ollama"
    name: str = "qwen2.5:3b-instruct"
    embed_model: str = "nomic-embed-text"
    embed_provider: Literal["ollama", "openai_compat", "gemini"] | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = 42
    max_tool_calls: int = Field(default=3, ge=0, le=10)


class AgentDefinition(BaseModel):
    """Complete declarative agent specification serializable to/from YAML."""

    meta: AgentMeta
    system_prompt: SystemPromptSections
    retrieval: AgentRetrievalSettings = Field(default_factory=AgentRetrievalSettings)
    tools: list[AttachedToolSpec] = Field(default_factory=list)
    output_contract: OutputContractConfig = Field(default_factory=OutputContractConfig)
    model: AgentModelSettings = Field(default_factory=AgentModelSettings)

    def attached_tool_names(self) -> list[str]:
        """Return ordered unique names of tools attached to this agent."""
        return [tool.name for tool in self.tools]

    def to_yaml(self) -> str:
        """Serialize this definition to a deterministic YAML string."""
        data = self.model_dump(mode="json")
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    def save_yaml(self, path: Path | str) -> Path:
        """Write this definition as YAML to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml(), encoding="utf-8")
        return target

    @classmethod
    def from_yaml(cls, yaml_text: str) -> AgentDefinition:
        """Parse and validate an AgentDefinition from YAML text."""
        loaded = yaml.safe_load(yaml_text)
        if not isinstance(loaded, dict):
            raise ValueError("AgentDefinition YAML must parse to a mapping/object.")
        return cls.model_validate(loaded)

    @classmethod
    def load_yaml(cls, path: Path | str) -> AgentDefinition:
        """Load and validate an AgentDefinition from a YAML file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Agent definition file not found: {target}")
        return cls.from_yaml(target.read_text(encoding="utf-8"))


class RetrievedChunkRecord(BaseModel):
    """Lightweight retrieval trace entry recorded in AnswerRecord."""

    chunk_id: str
    doc_id: str
    score: float
    page: int | None = None
    section_heading: str | None = None


class ToolCallTrace(BaseModel):
    """Execution record for a single tool invocation during runtime."""

    step: int
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class AnswerRecord(BaseModel):
    """Full execution record returned by AgentRuntime.answer()."""

    question: str
    output: AgentOutput
    retrieved_chunks: list[RetrievedChunkRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    effective_query: str = ""
    latency_ms: float = 0.0
    model: str = ""
    contract_repairs: int = 0
    contract_failed: bool = False
