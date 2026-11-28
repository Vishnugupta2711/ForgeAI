"""Generic declarative AgentRuntime executing any valid AgentDefinition."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from forge.agent.schema import (
    AgentDefinition,
    AgentOutput,
    AnswerRecord,
    RetrievedChunkRecord,
    ToolCallTrace,
)
from forge.config import ForgeConfig, load_config, load_prompt
from forge.llm.client import LLMClient, extract_json_value
from forge.llm.models import LLMMessage, LLMProviderError
from forge.retrieval.hybrid import HybridIndex, HybridRetriever, tokenize
from forge.retrieval.models import RetrievalConfig, SearchHit
from forge.tools import ToolContext, ToolRegistry, default_registry


class _RewriteQueryResult(BaseModel):
    rewritten_query: str = Field(min_length=1)


def _format_tools_block(
    definition: AgentDefinition, registry: ToolRegistry
) -> str:
    attached = definition.attached_tool_names()
    if not attached:
        return "None. Do not call any tools."
    specs = registry.list_specs(attached)
    if not specs:
        return "None registered."
    lines: list[str] = []
    for spec in specs:
        lines.append(
            f"- {spec.name}: {spec.description}\n"
            f"  args_schema: {json.dumps(spec.args_schema, separators=(',', ':'))}"
        )
    return "\n".join(lines)


def _format_history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "(none)"
    recent = history[-6:]
    lines = [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent]
    return "\n".join(lines)


def _format_context_block(hits: list[SearchHit]) -> str:
    if not hits:
        return "(No retrieved chunks passed the relevance threshold.)"
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        c = hit.chunk
        meta_parts = [f"doc_id={c.doc_id}", f"chunk_id={c.chunk_id}", f"score={hit.score:.4f}"]
        if c.page is not None:
            meta_parts.append(f"page={c.page}")
        if c.section_heading:
            meta_parts.append(f"section={c.section_heading}")
        blocks.append(f"[{idx}] ({' | '.join(meta_parts)})\n{c.text}")
    return "\n\n".join(blocks)


def _build_contract_schema_block(schema: dict[str, Any]) -> str:
    """Render the machine-generated output contract JSON schema regardless of format_rules."""
    props = sorted(schema.get("properties", {}).keys())
    cit_schema = schema.get("properties", {}).get("citations", {})
    cit_items = cit_schema.get("items", {}) if isinstance(cit_schema, dict) else {}
    if isinstance(cit_items, dict) and "$ref" in cit_items:
        ref_name = str(cit_items["$ref"]).rsplit("/", 1)[-1]
        cit_items = schema.get("$defs", {}).get(ref_name, {})
    cit_props = sorted(
        cit_items.get("properties", {}).keys() if isinstance(cit_items, dict) else []
    )
    schema_json = json.dumps(schema, indent=2)
    return (
        "Authoritative Output Contract Schema (always overrides conflicting format text):\n"
        f"- Required JSON top-level fields: {', '.join(props)}\n"
        f"- Required keys inside every `citations` item: {', '.join(cit_props)}\n"
        f"JSON Schema:\n{schema_json}"
    )


def _rerank_hits(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    q_terms = set(tokenize(query))
    if not q_terms:
        return hits

    def _key(hit: SearchHit) -> tuple[float, float]:
        c_terms = set(tokenize(f"{hit.chunk.section_heading or ''} {hit.chunk.text}"))
        overlap = len(q_terms & c_terms) / len(q_terms)
        return (overlap * 0.25 + hit.score, hit.score)

    return sorted(hits, key=_key, reverse=True)


class AgentRuntime:
    """Generic runtime that executes any declarative AgentDefinition over a HybridIndex."""

    def __init__(
        self,
        definition: AgentDefinition,
        index: HybridIndex,
        *,
        llm_client: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        tool_context: ToolContext | None = None,
        config: ForgeConfig | None = None,
    ) -> None:
        self.definition = definition
        self.index = index
        self.registry = registry or default_registry()
        self.tool_context = tool_context or ToolContext()
        self._allowed_tools = set(definition.attached_tool_names())

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            cfg = (config or load_config()).model_copy(deep=True)
            cfg.endpoints.provider = definition.model.provider
            cfg.endpoints.role_providers["agent"] = definition.model.provider
            if definition.model.embed_provider is not None:
                cfg.endpoints.role_providers["embed"] = definition.model.embed_provider
            cfg.models.agent = definition.model.name
            cfg.models.embed = definition.model.embed_model
            cfg.generation.temperature = definition.model.temperature
            cfg.generation.seed = definition.model.seed
            self.llm_client = LLMClient.from_config(cfg)

        # Merge tool configs from AgentDefinition into tool_context
        merged_tool_cfgs = dict(self.tool_context.tool_configs)
        for attached in definition.tools:
            if attached.config:
                existing = dict(merged_tool_cfgs.get(attached.name, {}))
                existing.update(attached.config)
                merged_tool_cfgs[attached.name] = existing
        self.tool_context = self.tool_context.model_copy(update={"tool_configs": merged_tool_cfgs})

    def _rewrite_query(self, question: str, trace: list[str]) -> str:
        prompt_tpl = load_prompt("agent_query_rewrite.v1.md")
        try:
            res = self.llm_client.json(
                prompt_tpl.format(question=question),
                schema=_RewriteQueryResult,
                model=self.definition.model.name,
                temperature=self.definition.model.temperature,
                seed=self.definition.model.seed,
                max_repairs=1,
                role="agent",
            )
            rewritten = res.rewritten_query.strip()
            if rewritten:
                trace.append(f"query_rewrite: '{question}' -> '{rewritten}'")
                return rewritten
        except Exception as exc:
            trace.append(f"query_rewrite failed ({exc}); using original question")
        return question

    def _retrieve(self, query: str, trace: list[str]) -> list[SearchHit]:
        use_dense = self.index.embeddings is not None and self.definition.retrieval.hybrid_alpha > 0
        alpha = self.definition.retrieval.hybrid_alpha if use_dense else 0.0
        retriever = HybridRetriever(
            self.index,
            llm_client=self.llm_client if use_dense else None,
            config=RetrievalConfig(
                alpha=alpha,
                top_k=self.definition.retrieval.top_k,
                min_score=None,
                query_rewrite=False,
            ),
        )
        try:
            hits = retriever.search(query)
        except LLMProviderError as exc:
            trace.append(f"dense retrieval unavailable ({exc}); falling back to BM25-only")
            hits = HybridRetriever(
                self.index,
                llm_client=None,
                config=RetrievalConfig(
                    alpha=0.0,
                    top_k=self.definition.retrieval.top_k,
                    min_score=None,
                ),
            ).search(query)

        trace.append(f"retrieved {len(hits)} raw chunks (alpha={alpha})")

        if self.definition.retrieval.min_score is not None:
            threshold = self.definition.retrieval.min_score
            before = len(hits)
            hits = [h for h in hits if retriever.normalize_score(h.score) >= threshold]
            trace.append(
                f"min_score gate ({threshold} normalized): kept {len(hits)}/{before} chunks"
            )

        if self.definition.retrieval.rerank and hits:
            hits = _rerank_hits(query, hits)
            trace.append("applied lexical-overlap rerank")

        return hits

    def _validate_contract(
        self,
        raw_content: str,
        conversation: list[LLMMessage],
        contract_json: str,
        trace: list[str],
        *,
        max_repairs: int = 2,
    ) -> tuple[AgentOutput, int, bool]:
        repair_tpl = load_prompt("agent_contract_repair.v1.md")
        current_raw = raw_content
        current_conv = list(conversation)
        repairs_used = 0

        for attempt in range(1, max_repairs + 2):
            try:
                parsed = extract_json_value(current_raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Final response must be a JSON object.")
                cleaned = {k: v for k, v in parsed.items() if k != "action"}
                return AgentOutput.model_validate(cleaned), repairs_used, False
            except (ValueError, ValidationError) as exc:
                err_msg = str(exc)
                trace.append(f"output_contract validation error (attempt {attempt}): {err_msg}")
                if repairs_used >= max_repairs:
                    break
                repairs_used += 1
                repair_prompt = repair_tpl.format(
                    error_message=err_msg,
                    output_contract_json=contract_json,
                )
                current_conv = [
                    *current_conv,
                    LLMMessage(role="assistant", content=current_raw),
                    LLMMessage(role="user", content=repair_prompt),
                ]
                repair_resp = self.llm_client.chat(
                    current_conv,
                    model=self.definition.model.name,
                    temperature=self.definition.model.temperature,
                    seed=self.definition.model.seed,
                    json_mode=True,
                    role="agent",
                    attempt=repairs_used + 1,
                )
                current_raw = repair_resp.content

        trace.append("output_contract repair exhausted; returning safe refusal fallback")
        return (
            AgentOutput(
                answer="Unable to produce a schema-compliant grounded response.",
                citations=[],
                refused=True,
                refusal_reason="Model output failed output contract validation.",
                confidence=0.0,
            ),
            repairs_used,
            True,
        )

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AnswerRecord:
        """Answer a user question using retrieval, bounded JSON tool loop, and contract check."""
        start = time.perf_counter()
        trace: list[str] = []
        tool_traces: list[ToolCallTrace] = []

        effective_query = (
            self._rewrite_query(question, trace)
            if self.definition.retrieval.query_rewrite
            else question
        )
        hits = self._retrieve(effective_query, trace)
        retrieved_records = [
            RetrievedChunkRecord(
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                score=h.score,
                page=h.chunk.page,
                section_heading=h.chunk.section_heading,
            )
            for h in hits
        ]

        contract_json = json.dumps(self.definition.output_contract.json_schema, indent=2)
        system_prompt = load_prompt("agent_runtime_system.v1.md").format(
            system_prompt_rendered=self.definition.system_prompt.render(),
            tools_block=_format_tools_block(self.definition, self.registry),
            output_contract_json=contract_json,
        )
        user_prompt = load_prompt("agent_runtime_user.v1.md").format(
            history_block=_format_history_block(history),
            context_block=_format_context_block(hits),
            question=question,
            contract_schema_block=_build_contract_schema_block(
                self.definition.output_contract.json_schema
            ),
        )
        conversation: list[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        obs_tpl = load_prompt("agent_tool_observation.v1.md")
        force_final_tpl = load_prompt("agent_force_final.v1.md")
        max_tools = self.definition.model.max_tool_calls
        executed_valid_tools = 0
        final_raw = ""

        for step in range(1, max_tools + 3):
            resp = self.llm_client.chat(
                conversation,
                model=self.definition.model.name,
                temperature=self.definition.model.temperature,
                seed=self.definition.model.seed,
                json_mode=True,
                role="agent",
            )
            final_raw = resp.content

            try:
                parsed = extract_json_value(resp.content)
            except ValueError:
                break

            if not isinstance(parsed, dict):
                break

            is_tool_call = parsed.get("action") == "tool_call" or (
                "tool_name" in parsed and "answer" not in parsed
            )
            if not is_tool_call:
                break

            tool_name = str(parsed.get("tool_name", "")).strip()
            raw_args = parsed.get("arguments") or {}
            if not isinstance(raw_args, dict):
                raw_args = {}

            if tool_name not in self._allowed_tools:
                err = (
                    f"Rejected unattached tool '{tool_name}'. "
                    f"Allowed tools: {sorted(self._allowed_tools)}"
                )
                trace.append(err)
                tool_traces.append(
                    ToolCallTrace(
                        step=step,
                        tool_name=tool_name,
                        arguments=raw_args,
                        ok=False,
                        error=err,
                    )
                )
                obs_msg = obs_tpl.format(
                    tool_name=tool_name,
                    ok=False,
                    result_json="null",
                    error=err,
                )
                conversation = [
                    *conversation,
                    LLMMessage(role="assistant", content=resp.content),
                    LLMMessage(role="user", content=obs_msg),
                ]
                continue

            if executed_valid_tools >= max_tools:
                trace.append(f"max_tool_calls ({max_tools}) reached; forcing final answer")
                conversation = [
                    *conversation,
                    LLMMessage(role="assistant", content=resp.content),
                    LLMMessage(
                        role="user",
                        content=force_final_tpl.format(
                            max_tool_calls=max_tools,
                            output_contract_json=contract_json,
                        ),
                    ),
                ]
                forced_resp = self.llm_client.chat(
                    conversation,
                    model=self.definition.model.name,
                    temperature=self.definition.model.temperature,
                    seed=self.definition.model.seed,
                    json_mode=True,
                    role="agent",
                )
                final_raw = forced_resp.content
                break

            t_start = time.perf_counter()
            tool_result = self.registry.call(
                tool_name,
                raw_args,
                ctx=self.tool_context,
                allowed_tools=self._allowed_tools,
            )
            t_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            executed_valid_tools += 1

            trace.append(
                f"tool_call[{executed_valid_tools}]: {tool_name} ok={tool_result.ok}"
            )
            tool_traces.append(
                ToolCallTrace(
                    step=step,
                    tool_name=tool_name,
                    arguments=raw_args,
                    ok=tool_result.ok,
                    result=tool_result.data,
                    error=tool_result.error,
                    latency_ms=t_ms,
                )
            )
            obs_msg = obs_tpl.format(
                tool_name=tool_name,
                ok=tool_result.ok,
                result_json=json.dumps(tool_result.data, ensure_ascii=False),
                error=tool_result.error or "none",
            )
            conversation = [
                *conversation,
                LLMMessage(role="assistant", content=resp.content),
                LLMMessage(role="user", content=obs_msg),
            ]

        output, contract_repairs, contract_failed = self._validate_contract(
            final_raw, conversation, contract_json, trace, max_repairs=2
        )
        total_ms = round((time.perf_counter() - start) * 1000.0, 2)

        return AnswerRecord(
            question=question,
            output=output,
            retrieved_chunks=retrieved_records,
            tool_calls=tool_traces,
            trace=trace,
            effective_query=effective_query,
            latency_ms=total_ms,
            model=self.definition.model.name,
            contract_repairs=contract_repairs,
            contract_failed=contract_failed,
        )
