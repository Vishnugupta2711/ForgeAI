"""Declarative AgentDefinition generator with prompt synthesis and runtime smoke test."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forge.agent.runtime import AgentRuntime
from forge.agent.schema import (
    AgentDefinition,
    AgentMeta,
    AgentModelSettings,
    AgentProvenance,
    OutputContractConfig,
    SystemPromptSections,
    default_output_contract_schema,
)
from forge.config import ForgeConfig, load_prompt
from forge.generator.heuristics import derive_retrieval_settings
from forge.generator.tool_gate import ToolDecision, gate_tools
from forge.ingest.models import CorpusProfile, Document
from forge.intake.models import RequirementsDocument
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.retrieval.hybrid import HybridIndex
from forge.tools import ToolContext, ToolRegistry, default_registry

_GENERATE_PROMPT_FILES = [
    "generate_tool_proposals.v1.md",
    "generate_role.v1.md",
    "generate_scope.v1.md",
    "generate_audience_tone.v1.md",
    "generate_grounding_rules.v1.md",
    "generate_refusal_policy.v1.md",
    "generate_format_rules.v1.md",
    "generate_extra_rules.v1.md",
]


class _ToolProposalItem(BaseModel):
    name: str
    reason: str = ""


class _ToolProposalList(BaseModel):
    proposed_tools: list[_ToolProposalItem] = Field(default_factory=list)


class _RoleOutput(BaseModel):
    role: str = Field(min_length=1)


class _ScopeOutput(BaseModel):
    scope: str = Field(min_length=1)


class _AudienceToneOutput(BaseModel):
    audience_tone: str = Field(min_length=1)


class _GroundingRulesOutput(BaseModel):
    grounding_rules: str = Field(min_length=1)


class _RefusalPolicyOutput(BaseModel):
    refusal_policy: str = Field(min_length=1)


class _FormatRulesOutput(BaseModel):
    format_rules: str = Field(min_length=1)


class _ExtraRulesOutput(BaseModel):
    extra_rules: list[str] = Field(default_factory=list)


class SmokeTestCaseResult(BaseModel):
    """Result from executing a generated agent on one smoke-test question."""

    question: str
    expected_refusal: bool = False
    refused: bool
    citations_count: int
    tool_calls: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    answer_preview: str = ""
    ok: bool = True
    error: str | None = None


class GenerationLog(BaseModel):
    """Audit artifact recording heuristics, tool gating, and smoke-test verification."""

    agent_name: str
    brief: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    retrieval_rationale: dict[str, Any] = Field(default_factory=dict)
    tool_decisions: list[ToolDecision] = Field(default_factory=list)
    smoke_test_results: list[SmokeTestCaseResult] = Field(default_factory=list)
    llm_usage_summary: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        """Write generation_log.json to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target


def _slugify_name(brief: str) -> str:
    words = re.findall(r"[a-z0-9]+", brief.lower())
    stop = {"i", "need", "an", "a", "the", "that", "for", "agent"}
    filtered = [w for w in words if w not in stop]
    slug = "-".join((filtered or words)[:4])
    return f"{slug}-assistant" if slug else "generated-assistant"


def _compute_prompt_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for fname in _GENERATE_PROMPT_FILES:
        content = load_prompt(fname)
        hashes[fname] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return hashes


def _build_output_contract(requirements: RequirementsDocument) -> OutputContractConfig:
    schema = default_output_contract_schema()
    schema["description"] = (
        f"Output contract for scope: {requirements.scope[:180]}. "
        "Citations with verbatim quotes are required whenever refused is false."
    )
    props = schema.get("properties", {})
    if "citations" in props:
        props["citations"]["description"] = (
            "Supporting citations (doc_id, chunk_id, verbatim quote). "
            "Required and non-empty whenever refused is false."
        )
    if "refused" in props:
        props["refused"]["description"] = (
            "True when the question falls under must-refuse topics or lacks document support."
        )
    return OutputContractConfig(json_schema=schema)


def _derive_smoke_questions(requirements: RequirementsDocument) -> list[tuple[str, bool]]:
    questions: list[tuple[str, bool]] = []
    for topic in requirements.must_answer_topics[:2]:
        questions.append((f"Explain {topic.title}: {topic.description}", False))

    while len(questions) < 2:
        questions.append((f"Summarize what is documented regarding: {requirements.scope}", False))

    if requirements.must_refuse_topics:
        ref = requirements.must_refuse_topics[0]
        questions.append((f"Provide details about {ref.title} ({ref.reason}).", True))
    else:
        questions.append(
            ("Provide confidential internal financial projections not in the documents.", True)
        )
    return questions[:3]


class AgentGenerator:
    """Generate a declarative AgentDefinition + GenerationLog from requirements & profile."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        registry: ToolRegistry | None = None,
        config: ForgeConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.registry = registry or default_registry()
        self.config = config or llm_client.config

    def _propose_tools(
        self, requirements: RequirementsDocument, profile: CorpusProfile
    ) -> list[str]:
        caps_block = "\n".join(
            f"- {c.name} (linked={c.linked_req_ids}): {c.reason}"
            for c in requirements.capabilities_needed
        ) or "(none)"
        tools_block = "\n".join(
            f"- {s.name}: {s.description} (tags={s.capability_tags})"
            for s in self.registry.list_specs()
        )
        prompt = load_prompt("generate_tool_proposals.v1.md").format(
            brief=requirements.brief,
            capabilities_block=caps_block,
            corpus_profile=profile.to_compact_text(),
            tools_block=tools_block,
        )
        try:
            parsed = self.llm_client.json(
                prompt,
                schema=_ToolProposalList,
                role="agent",
                temperature=0.0,
                seed=self.config.generation.seed,
                max_repairs=2,
            )
            return [p.name.strip() for p in parsed.proposed_tools if p.name.strip()]
        except LLMProviderError:
            raise
        except Exception:
            return []

    def _generate_sections(
        self,
        requirements: RequirementsDocument,
        profile: CorpusProfile,
        attached_tool_names: list[str],
    ) -> SystemPromptSections:
        must_answer_block = "\n".join(
            f"- {t.id}: {t.title} — {t.description}"
            for t in requirements.must_answer_topics
        ) or "(none)"
        must_refuse_block = "\n".join(
            f"- {r.id}: {r.title} — {r.reason}"
            for r in requirements.must_refuse_topics
        ) or "(none)"
        gaps_block = "\n".join(
            f"- {g}"
            for g in (
                requirements.groundability_report.flagged_gaps
                if requirements.groundability_report
                else []
            )
        ) or "(none)"
        assumptions_block = "\n".join(f"- {a}" for a in requirements.assumptions) or "(none)"
        tools_str = ", ".join(attached_tool_names) if attached_tool_names else "none"

        def _call_section(fname: str, schema: type[BaseModel], kwargs: dict[str, Any]) -> Any:
            prompt = load_prompt(fname).format(**kwargs)
            return self.llm_client.json(
                prompt,
                schema=schema,
                role="agent",
                temperature=0.0,
                seed=self.config.generation.seed,
                max_repairs=2,
            )

        try:
            role_text = _call_section(
                "generate_role.v1.md",
                _RoleOutput,
                {
                    "brief": requirements.brief,
                    "scope": requirements.scope,
                    "audience": requirements.audience,
                    "tone": requirements.tone,
                    "must_answer_block": must_answer_block,
                    "must_refuse_block": must_refuse_block,
                },
            ).role.strip()
        except LLMProviderError:
            raise
        except Exception:
            grounded_summary = (
                ", ".join(t.title for t in requirements.must_answer_topics)
                if requirements.must_answer_topics
                else requirements.scope
            )
            role_text = (
                f"You are a grounded assistant for {requirements.audience}. "
                f"Answer strictly from supplied documents regarding: {grounded_summary}."
            )

        try:
            scope_text = _call_section(
                "generate_scope.v1.md",
                _ScopeOutput,
                {
                    "scope": requirements.scope,
                    "must_answer_block": must_answer_block,
                    "must_refuse_block": must_refuse_block,
                },
            ).scope.strip()
        except LLMProviderError:
            raise
        except Exception:
            scope_text = requirements.scope

        try:
            audience_tone_text = _call_section(
                "generate_audience_tone.v1.md",
                _AudienceToneOutput,
                {
                    "audience": requirements.audience,
                    "tone": requirements.tone,
                    "assumptions_block": assumptions_block,
                },
            ).audience_tone.strip()
        except LLMProviderError:
            raise
        except Exception:
            audience_tone_text = (
                f"Target audience: {requirements.audience}. Tone: {requirements.tone}."
            )

        try:
            grounding_text = _call_section(
                "generate_grounding_rules.v1.md",
                _GroundingRulesOutput,
                {
                    "attached_tools": tools_str,
                    "corpus_profile": profile.to_compact_text(),
                },
            ).grounding_rules.strip()
        except LLMProviderError:
            raise
        except Exception:
            grounding_text = (
                "Cite verbatim quoted evidence (doc_id, chunk_id, quote) from retrieved chunks "
                "for every factual claim. Never fabricate facts."
            )

        # Deterministic check: grounding_rules MUST require citing quoted evidence
        lower_ground = grounding_text.lower()
        if "quote" not in lower_ground or "cit" not in lower_ground:
            grounding_text = (
                f"{grounding_text} Always cite verbatim quoted evidence "
                "(doc_id, chunk_id, quote) from retrieved chunks for every claim."
            )

        try:
            refusal_text = _call_section(
                "generate_refusal_policy.v1.md",
                _RefusalPolicyOutput,
                {
                    "must_refuse_block": must_refuse_block,
                    "gaps_block": gaps_block,
                },
            ).refusal_policy.strip()
        except LLMProviderError:
            raise
        except Exception:
            refusal_text = (
                "If a question asks about excluded topics or unsupported gaps, "
                "set refused=true and state honestly that the documents don't cover this."
            )

        # Deterministic check: refusal_policy must include must_refuse topics, gaps, & honest phrase
        missing_refusals = [
            r.title for r in requirements.must_refuse_topics
            if r.title.lower() not in refusal_text.lower()
        ]
        if missing_refusals:
            refusal_text = (
                f"{refusal_text} Excluded/ungrounded topics to refuse: "
                + "; ".join(missing_refusals)
                + "."
            )
        if "don't cover" not in refusal_text.lower() and "do not cover" not in refusal_text.lower():
            refusal_text = (
                f"{refusal_text} When evidence is missing, refuse honestly stating that "
                "the documents don't cover this."
            )

        try:
            format_text = _call_section(
                "generate_format_rules.v1.md",
                _FormatRulesOutput,
                {"attached_tools": tools_str},
            ).format_rules.strip()
        except LLMProviderError:
            raise
        except Exception:
            format_text = (
                "Return valid JSON matching the output contract: answer, citations "
                "(required whenever refused is false), refused, refusal_reason, confidence."
            )

        try:
            extra_rules = _call_section(
                "generate_extra_rules.v1.md",
                _ExtraRulesOutput,
                {
                    "assumptions_block": assumptions_block,
                    "attached_tools": tools_str,
                    "must_answer_block": must_answer_block,
                },
            ).extra_rules
            extra_rules = [r.strip() for r in extra_rules if r.strip()][:5]
        except LLMProviderError:
            raise
        except Exception:
            extra_rules = [
                "Use only attached tools listed in this definition.",
                "Do not speculate beyond retrieved document passages.",
            ]

        return SystemPromptSections(
            role=role_text,
            scope=scope_text,
            audience_tone=audience_tone_text,
            grounding_rules=grounding_text,
            refusal_policy=refusal_text,
            format_rules=format_text,
            extra_rules=extra_rules,
        )

    def _run_smoke_tests(
        self,
        definition: AgentDefinition,
        index: HybridIndex,
        requirements: RequirementsDocument,
        documents: list[Document] | None,
    ) -> list[SmokeTestCaseResult]:
        tool_ctx = (
            ToolContext.from_documents(documents, offline_only=True)
            if documents is not None
            else ToolContext(offline_only=True)
        )
        runtime = AgentRuntime(
            definition,
            index,
            llm_client=self.llm_client,
            registry=self.registry,
            tool_context=tool_ctx,
            config=self.config,
        )
        results: list[SmokeTestCaseResult] = []
        for question, expected_refusal in _derive_smoke_questions(requirements):
            try:
                rec = runtime.answer(question)
                results.append(
                    SmokeTestCaseResult(
                        question=question,
                        expected_refusal=expected_refusal,
                        refused=rec.output.refused,
                        citations_count=len(rec.output.citations),
                        tool_calls=[tc.tool_name for tc in rec.tool_calls],
                        latency_ms=rec.latency_ms,
                        answer_preview=rec.output.answer[:160],
                        ok=True,
                    )
                )
            except Exception as exc:
                results.append(
                    SmokeTestCaseResult(
                        question=question,
                        expected_refusal=expected_refusal,
                        refused=False,
                        citations_count=0,
                        ok=False,
                        error=str(exc),
                    )
                )
        return results

    def generate(
        self,
        requirements: RequirementsDocument,
        profile: CorpusProfile,
        index: HybridIndex,
        *,
        documents: list[Document] | None = None,
        agent_name: str | None = None,
        out_dir: Path | str | None = None,
        run_smoke_test: bool = True,
    ) -> tuple[AgentDefinition, GenerationLog]:
        """Generate an AgentDefinition and GenerationLog from requirements + profile."""
        prompt_hashes = _compute_prompt_hashes()
        retrieval_settings, retrieval_rationale = derive_retrieval_settings(profile, requirements)

        llm_proposed = self._propose_tools(requirements, profile)
        attached_tools, tool_decisions = gate_tools(
            requirements,
            profile,
            self.registry,
            llm_proposed=llm_proposed,
        )
        attached_names = [t.name for t in attached_tools]

        sections = self._generate_sections(requirements, profile, attached_names)
        output_contract = _build_output_contract(requirements)
        resolved_name = agent_name or _slugify_name(requirements.brief)

        definition = AgentDefinition(
            meta=AgentMeta(
                name=resolved_name,
                brief=requirements.brief,
                version="1.0.0",
                provenance=AgentProvenance(
                    model_names={
                        "agent": self.config.models.agent,
                        "judge": self.config.models.judge,
                        "eval_writer": self.config.models.eval_writer,
                        "embed": self.config.models.embed,
                    },
                    prompt_hashes=prompt_hashes,
                    generator_version="0.1.0",
                ),
            ),
            system_prompt=sections,
            retrieval=retrieval_settings,
            tools=attached_tools,
            output_contract=output_contract,
            model=AgentModelSettings(
                provider=self.config.endpoints.provider,
                name=self.config.models.agent,
                embed_model=self.config.models.embed,
                temperature=self.config.generation.temperature,
                seed=self.config.generation.seed,
                max_tool_calls=3,
            ),
        )

        smoke_results: list[SmokeTestCaseResult] = []
        if run_smoke_test:
            smoke_results = self._run_smoke_tests(
                definition, index, requirements, documents
            )

        log = GenerationLog(
            agent_name=resolved_name,
            brief=requirements.brief,
            prompt_hashes=prompt_hashes,
            retrieval_rationale=retrieval_rationale,
            tool_decisions=tool_decisions,
            smoke_test_results=smoke_results,
            llm_usage_summary=self.llm_client.usage_summary(),
        )

        if out_dir is not None:
            target = Path(out_dir)
            target.mkdir(parents=True, exist_ok=True)
            definition.save_yaml(target / "definition.yaml")
            log.save(target / "generation_log.json")

        return definition, log
