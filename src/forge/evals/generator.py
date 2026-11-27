"""Source-verified evaluation dataset generator."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from pydantic import BaseModel, Field

from forge.agent.schema import AgentDefinition
from forge.config import ForgeConfig, load_prompt
from forge.evals.gates import (
    check_oos_unanswerable,
    check_solvability,
    dedupe_by_embedding,
    stratified_split,
    verify_case_key_facts,
)
from forge.evals.models import (
    EvalCase,
    EvalCategory,
    EvalProvenance,
    EvalSet,
    EvalSourceRef,
    ExpectedBehavior,
    RejectedCaseLog,
)
from forge.ingest.models import Chunk, CorpusProfile
from forge.intake.models import RequirementsDocument
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.retrieval.hybrid import HybridIndex


class _SingleDocRawItem(BaseModel):
    question: str
    key_facts: list[str] = Field(default_factory=list)
    notes: str = ""


class _SingleDocRawList(BaseModel):
    cases: list[_SingleDocRawItem] = Field(default_factory=list)


class _MultiDocRawItem(BaseModel):
    question: str
    key_facts: list[str] = Field(default_factory=list)
    notes: str = ""


class _MultiDocRawList(BaseModel):
    cases: list[_MultiDocRawItem] = Field(default_factory=list)


class _OosRawItem(BaseModel):
    question: str
    subtype: str = "out_of_scope"
    notes: str = ""


class _OosRawList(BaseModel):
    cases: list[_OosRawItem] = Field(default_factory=list)


class _EdgeRawItem(BaseModel):
    question: str
    expected_behavior: ExpectedBehavior = "answer"
    source_chunk_ids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    subtype: str = "edge_case"
    notes: str = ""


class _EdgeRawList(BaseModel):
    cases: list[_EdgeRawItem] = Field(default_factory=list)


class _ToolRawItem(BaseModel):
    question: str
    target_tool: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    notes: str = ""


class _ToolRawList(BaseModel):
    cases: list[_ToolRawItem] = Field(default_factory=list)


def model_family(model_name: str) -> str:
    """Extract normalized model family prefix (e.g. 'qwen2.5:3b' -> 'qwen2.5')."""
    base = model_name.strip().lower().split("/", 1)[-1]
    base = base.split(":", 1)[0]
    for sep in ("-", "_"):
        if sep in base:
            return base.split(sep, 1)[0]
    return base


def default_category_quotas(has_tools: bool) -> dict[EvalCategory, int]:
    """Return default category quotas totaling 36 cases."""
    if has_tools:
        return {
            "single_doc": 12,
            "multi_doc": 6,
            "out_of_scope": 8,
            "edge_case": 6,
            "tool_use": 4,
        }
    return {
        "single_doc": 14,
        "multi_doc": 7,
        "out_of_scope": 9,
        "edge_case": 6,
        "tool_use": 0,
    }


def _prompt_hash(filename: str) -> str:
    return hashlib.sha256(load_prompt(filename).encode("utf-8")).hexdigest()[:16]


class EvalGenerator:
    """Generate and gate a source-verified dev/held_out EvalSet from documents."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        config: ForgeConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or llm_client.config

    def _eval_model(self) -> str:
        return self.config.models.eval_writer or self.config.models.judge

    def _make_provenance(self, prompt_file: str, seed: int) -> EvalProvenance:
        return EvalProvenance(
            prompt_file=prompt_file,
            prompt_hash=_prompt_hash(prompt_file),
            model=self._eval_model(),
            seed=seed,
        )

    def _validate_and_filter(
        self,
        candidates: list[EvalCase],
        chunk_map: dict[str, Chunk],
        index: HybridIndex,
        *,
        corpus_doc_count: int,
        seed: int,
    ) -> tuple[list[EvalCase], list[RejectedCaseLog]]:
        accepted: list[EvalCase] = []
        rejected: list[RejectedCaseLog] = []
        model = self._eval_model()

        for case in candidates:
            # Gate 1: key_fact verification
            verified_case, g1_err = verify_case_key_facts(
                case,
                chunk_map,
                require_multi_doc=(case.category == "multi_doc"),
                corpus_doc_count=corpus_doc_count,
            )
            if verified_case is None:
                rejected.append(
                    RejectedCaseLog(
                        question=case.question,
                        category=case.category,
                        gate="key_fact_verification",
                        reason=g1_err or "failed_key_fact_check",
                    )
                )
                continue

            # Gate 2: solvability check for answerable cases
            if verified_case.expected_behavior == "answer":
                solvable, reason = check_solvability(
                    verified_case, chunk_map, self.llm_client, model=model, seed=seed
                )
                if not solvable:
                    rejected.append(
                        RejectedCaseLog(
                            question=verified_case.question,
                            category=verified_case.category,
                            gate="solvability_check",
                            reason=reason or "not_solvable_from_gold_chunks",
                        )
                    )
                    continue

            # Gate 3: OOS unanswerability check
            if verified_case.expected_behavior == "refuse":
                is_oos, reason = check_oos_unanswerable(
                    verified_case, index, self.llm_client, model=model, seed=seed
                )
                if not is_oos:
                    rejected.append(
                        RejectedCaseLog(
                            question=verified_case.question,
                            category=verified_case.category,
                            gate="oos_answerability_check",
                            reason=reason or "covered_by_documents",
                        )
                    )
                    continue

            accepted.append(verified_case)

        return accepted, rejected

    def _generate_single_doc_batch(
        self,
        chunks: list[Chunk],
        requirements: RequirementsDocument,
        count_per_chunk: int,
        *,
        seed: int,
    ) -> list[EvalCase]:
        prompt_file = "evalgen_single_doc.v1.md"
        tpl = load_prompt(prompt_file)
        prov = self._make_provenance(prompt_file, seed)
        out: list[EvalCase] = []

        for chunk in chunks:
            prompt = tpl.format(
                count=count_per_chunk,
                scope=requirements.scope,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
            )
            try:
                res = self.llm_client.json(
                    prompt,
                    schema=_SingleDocRawList,
                    model=self._eval_model(),
                    temperature=0.0,
                    seed=seed,
                    role="eval_writer",
                    max_repairs=2,
                )
            except LLMProviderError:
                raise
            except Exception:
                continue

            for idx, item in enumerate(res.cases, start=1):
                if not item.question.strip():
                    continue
                out.append(
                    EvalCase(
                        id=f"CAND-S-{len(out) + idx}",
                        category="single_doc",
                        question=item.question.strip(),
                        expected_behavior="answer",
                        expected_sources=[
                            EvalSourceRef(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id)
                        ],
                        key_facts=list(item.key_facts),
                        notes=item.notes,
                        provenance=prov,
                    )
                )
        return out

    def _generate_multi_doc_batch(
        self,
        chunk_pairs: list[tuple[Chunk, Chunk]],
        *,
        count_per_pair: int,
        seed: int,
    ) -> list[EvalCase]:
        prompt_file = "evalgen_multi_doc.v1.md"
        tpl = load_prompt(prompt_file)
        prov = self._make_provenance(prompt_file, seed)
        out: list[EvalCase] = []

        for ch_a, ch_b in chunk_pairs:
            prompt = tpl.format(
                count=count_per_pair,
                doc_id_a=ch_a.doc_id,
                chunk_id_a=ch_a.chunk_id,
                chunk_text_a=ch_a.text,
                doc_id_b=ch_b.doc_id,
                chunk_id_b=ch_b.chunk_id,
                chunk_text_b=ch_b.text,
            )
            try:
                res = self.llm_client.json(
                    prompt,
                    schema=_MultiDocRawList,
                    model=self._eval_model(),
                    temperature=0.0,
                    seed=seed,
                    role="eval_writer",
                    max_repairs=2,
                )
            except LLMProviderError:
                raise
            except Exception:
                continue

            for idx, item in enumerate(res.cases, start=1):
                if not item.question.strip():
                    continue
                out.append(
                    EvalCase(
                        id=f"CAND-M-{len(out) + idx}",
                        category="multi_doc",
                        question=item.question.strip(),
                        expected_behavior="answer",
                        expected_sources=[
                            EvalSourceRef(doc_id=ch_a.doc_id, chunk_id=ch_a.chunk_id),
                            EvalSourceRef(doc_id=ch_b.doc_id, chunk_id=ch_b.chunk_id),
                        ],
                        key_facts=list(item.key_facts),
                        notes=item.notes,
                        provenance=prov,
                    )
                )
        return out

    def _generate_oos_batch(
        self,
        requirements: RequirementsDocument,
        profile: CorpusProfile,
        *,
        count: int,
        seed: int,
    ) -> list[EvalCase]:
        prompt_file = "evalgen_out_of_scope.v1.md"
        tpl = load_prompt(prompt_file)
        prov = self._make_provenance(prompt_file, seed)

        must_refuse_block = "\n".join(
            f"- {r.id}: {r.title} — {r.reason}" for r in requirements.must_refuse_topics
        ) or "(none)"
        gaps_block = "\n".join(
            f"- {g}"
            for g in (
                requirements.groundability_report.flagged_gaps
                if requirements.groundability_report
                else []
            )
        ) or "(none)"

        prompt = tpl.format(
            count=count,
            scope=requirements.scope,
            must_refuse_block=must_refuse_block,
            gaps_block=gaps_block,
            corpus_profile=profile.to_compact_text(),
        )
        try:
            res = self.llm_client.json(
                prompt,
                schema=_OosRawList,
                model=self._eval_model(),
                temperature=0.0,
                seed=seed,
                role="eval_writer",
                max_repairs=2,
            )
        except LLMProviderError:
            raise
        except Exception:
            return []

        out: list[EvalCase] = []
        for idx, item in enumerate(res.cases, start=1):
            if not item.question.strip():
                continue
            out.append(
                EvalCase(
                    id=f"CAND-O-{idx}",
                    category="out_of_scope",
                    question=item.question.strip(),
                    expected_behavior="refuse",
                    expected_sources=[],
                    key_facts=[],
                    notes=f"[{item.subtype}] {item.notes}".strip(),
                    provenance=prov,
                )
            )
        return out

    def _generate_edge_batch(
        self,
        chunks: list[Chunk],
        chunk_map: dict[str, Chunk],
        requirements: RequirementsDocument,
        *,
        count: int,
        seed: int,
    ) -> list[EvalCase]:
        prompt_file = "evalgen_edge_cases.v1.md"
        tpl = load_prompt(prompt_file)
        prov = self._make_provenance(prompt_file, seed)

        excerpts = "\n\n".join(
            f"[doc_id={c.doc_id} | chunk_id={c.chunk_id}]\n{c.text}" for c in chunks[:6]
        )
        prompt = tpl.format(
            count=count,
            scope=requirements.scope,
            excerpts_block=excerpts,
        )
        try:
            res = self.llm_client.json(
                prompt,
                schema=_EdgeRawList,
                model=self._eval_model(),
                temperature=0.0,
                seed=seed,
                role="eval_writer",
                max_repairs=2,
            )
        except LLMProviderError:
            raise
        except Exception:
            return []

        out: list[EvalCase] = []
        for idx, item in enumerate(res.cases, start=1):
            if not item.question.strip():
                continue
            refs: list[EvalSourceRef] = []
            for cid in item.source_chunk_ids:
                if cid in chunk_map:
                    refs.append(EvalSourceRef(doc_id=chunk_map[cid].doc_id, chunk_id=cid))
            if not refs and chunks:
                refs = [EvalSourceRef(doc_id=chunks[0].doc_id, chunk_id=chunks[0].chunk_id)]

            out.append(
                EvalCase(
                    id=f"CAND-E-{idx}",
                    category="edge_case",
                    question=item.question.strip(),
                    expected_behavior=item.expected_behavior,
                    expected_sources=refs if item.expected_behavior != "refuse" else [],
                    key_facts=list(item.key_facts),
                    notes=f"[{item.subtype}] {item.notes}".strip(),
                    provenance=prov,
                )
            )
        return out

    def _generate_tool_batch(
        self,
        definition: AgentDefinition,
        chunks: list[Chunk],
        chunk_map: dict[str, Chunk],
        *,
        count: int,
        seed: int,
    ) -> list[EvalCase]:
        if not definition.tools or count <= 0:
            return []

        prompt_file = "evalgen_tool_cases.v1.md"
        tpl = load_prompt(prompt_file)
        prov = self._make_provenance(prompt_file, seed)

        tools_block = "\n".join(
            f"- {t.name} (req={t.justification_requirement_id})" for t in definition.tools
        )
        csv_first = [c for c in chunks if c.metadata.get("doc_type") == "csv"]
        selected_chunks = (csv_first[:4] + [c for c in chunks if c not in csv_first][:4])[:6]
        excerpts = "\n\n".join(
            f"[doc_id={c.doc_id} | chunk_id={c.chunk_id}]\n{c.text}" for c in selected_chunks
        )
        prompt = tpl.format(
            count=count,
            tools_block=tools_block,
            excerpts_block=excerpts,
        )
        try:
            res = self.llm_client.json(
                prompt,
                schema=_ToolRawList,
                model=self._eval_model(),
                temperature=0.0,
                seed=seed,
                role="eval_writer",
                max_repairs=2,
            )
        except LLMProviderError:
            raise
        except Exception:
            return []

        out: list[EvalCase] = []
        for idx, item in enumerate(res.cases, start=1):
            if not item.question.strip():
                continue
            refs: list[EvalSourceRef] = []
            for cid in item.source_chunk_ids:
                if cid in chunk_map:
                    refs.append(EvalSourceRef(doc_id=chunk_map[cid].doc_id, chunk_id=cid))
            if not refs and selected_chunks:
                refs = [
                    EvalSourceRef(
                        doc_id=selected_chunks[0].doc_id,
                        chunk_id=selected_chunks[0].chunk_id,
                    )
                ]

            out.append(
                EvalCase(
                    id=f"CAND-T-{idx}",
                    category="tool_use",
                    question=item.question.strip(),
                    expected_behavior="answer",
                    expected_sources=refs,
                    key_facts=list(item.key_facts),
                    notes=f"[tool:{item.target_tool}] {item.notes}".strip(),
                    provenance=prov,
                )
            )
        return out

    def generate(
        self,
        definition: AgentDefinition,
        requirements: RequirementsDocument,
        profile: CorpusProfile,
        index: HybridIndex,
        *,
        quotas: dict[EvalCategory, int] | None = None,
        seed: int | None = None,
        out_dir: Path | str | None = None,
        max_rounds: int = 3,
    ) -> EvalSet:
        """Generate, verify, deduplicate, and split an EvalSet from corpus chunks."""
        resolved_seed = self.config.generation.seed if seed is None else seed
        chunks = list(index.chunks)
        chunk_map = {c.chunk_id: c for c in chunks}
        corpus_docs = {c.doc_id for c in chunks}
        rng = random.Random(resolved_seed)

        # Build chunk pairs across distinct docs for multi_doc generation
        doc_buckets: dict[str, list[Chunk]] = {}
        for ch in chunks:
            doc_buckets.setdefault(ch.doc_id, []).append(ch)
        doc_names = sorted(doc_buckets)
        chunk_pairs: list[tuple[Chunk, Chunk]] = []
        if len(doc_names) >= 2:
            for i in range(len(doc_names)):
                for j in range(i + 1, len(doc_names)):
                    b_a = doc_buckets[doc_names[i]]
                    b_b = doc_buckets[doc_names[j]]
                    cand_a = [c for c in b_a if len(c.text.strip()) >= 120] or b_a
                    cand_b = [c for c in b_b if len(c.text.strip()) >= 120] or b_b
                    for idx_a in range(min(4, len(cand_a))):
                        chunk_a = cand_a[(idx_a * 3) % len(cand_a)]
                        chunk_b = cand_b[(idx_a * 5 + 1) % len(cand_b)]
                        chunk_pairs.append((chunk_a, chunk_b))
        elif len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                chunk_pairs.append((chunks[i], chunks[i + 1]))

        has_tools = len(definition.tools) > 0
        scaled_note: str | None = None
        if quotas is not None:
            target_quotas = dict(quotas)
        else:
            base_quotas = default_category_quotas(has_tools)
            total_chars = sum(len(c.text) for c in chunks)
            if len(chunks) < 5 or total_chars < 1000:
                target_quotas = {
                    "single_doc": min(base_quotas["single_doc"], max(2, len(chunks) * 2)),
                    "multi_doc": min(base_quotas["multi_doc"], len(chunk_pairs) * 2),
                    "out_of_scope": min(
                        base_quotas["out_of_scope"],
                        max(3, len(requirements.must_refuse_topics) + 2),
                    ),
                    "edge_case": min(base_quotas["edge_case"], max(2, len(chunks))),
                    "tool_use": min(base_quotas["tool_use"], 2) if has_tools else 0,
                }
                scaled_note = (
                    f"Short corpus ({len(chunks)} chunks, {total_chars} chars): "
                    f"eval quotas scaled gracefully from {sum(base_quotas.values())} "
                    f"to {sum(target_quotas.values())} cases."
                )
            else:
                target_quotas = base_quotas
        if not has_tools:
            target_quotas["tool_use"] = 0

        accepted_by_cat: dict[EvalCategory, list[EvalCase]] = {
            "single_doc": [],
            "multi_doc": [],
            "out_of_scope": [],
            "edge_case": [],
            "tool_use": [],
        }
        all_rejected: list[RejectedCaseLog] = []

        for round_idx in range(max_rounds):
            round_seed = resolved_seed + round_idx
            batch_candidates: list[EvalCase] = []

            # 1. single_doc
            need_single = target_quotas.get("single_doc", 0) - len(accepted_by_cat["single_doc"])
            if need_single > 0 and chunks:
                sampled_chunks = list(chunks)
                rng.shuffle(sampled_chunks)
                batch_chunks = sampled_chunks[: max(1, min(len(sampled_chunks), need_single + 2))]
                per_chunk = max(1, (need_single + len(batch_chunks) - 1) // len(batch_chunks))
                batch_candidates.extend(
                    self._generate_single_doc_batch(
                        batch_chunks, requirements, per_chunk, seed=round_seed
                    )
                )

            # 2. multi_doc
            need_multi = target_quotas.get("multi_doc", 0) - len(accepted_by_cat["multi_doc"])
            if need_multi > 0 and chunk_pairs:
                sampled_pairs = list(chunk_pairs)
                rng.shuffle(sampled_pairs)
                batch_pairs = sampled_pairs[: max(1, min(len(sampled_pairs), need_multi + 1))]
                per_pair = max(1, (need_multi + len(batch_pairs) - 1) // len(batch_pairs))
                batch_candidates.extend(
                    self._generate_multi_doc_batch(
                        batch_pairs, count_per_pair=per_pair, seed=round_seed
                    )
                )

            # 3. out_of_scope
            need_oos = target_quotas.get("out_of_scope", 0) - len(accepted_by_cat["out_of_scope"])
            if need_oos > 0:
                batch_candidates.extend(
                    self._generate_oos_batch(
                        requirements, profile, count=need_oos + 2, seed=round_seed
                    )
                )

            # 4. edge_case
            need_edge = target_quotas.get("edge_case", 0) - len(accepted_by_cat["edge_case"])
            if need_edge > 0 and chunks:
                batch_candidates.extend(
                    self._generate_edge_batch(
                        chunks, chunk_map, requirements, count=need_edge + 2, seed=round_seed
                    )
                )

            # 5. tool_use
            need_tool = target_quotas.get("tool_use", 0) - len(accepted_by_cat["tool_use"])
            if need_tool > 0 and has_tools and chunks:
                batch_candidates.extend(
                    self._generate_tool_batch(
                        definition, chunks, chunk_map, count=need_tool + 2, seed=round_seed
                    )
                )

            if not batch_candidates:
                break

            verified, round_rejected = self._validate_and_filter(
                batch_candidates,
                chunk_map,
                index,
                corpus_doc_count=len(corpus_docs),
                seed=round_seed,
            )
            all_rejected.extend(round_rejected)

            # Dedupe across already accepted + new verified
            combined_pool = [
                *[c for cat_list in accepted_by_cat.values() for c in cat_list],
                *verified,
            ]
            deduped, dedupe_rejected = dedupe_by_embedding(combined_pool, self.llm_client)
            all_rejected.extend(dedupe_rejected)

            # Refill accepted_by_cat respecting quotas
            rebuilt: dict[EvalCategory, list[EvalCase]] = {
                "single_doc": [],
                "multi_doc": [],
                "out_of_scope": [],
                "edge_case": [],
                "tool_use": [],
            }
            for item in deduped:
                quota = target_quotas.get(item.category, 0)
                if len(rebuilt[item.category]) < quota:
                    rebuilt[item.category].append(item)
            accepted_by_cat = rebuilt

            if all(
                len(accepted_by_cat[cat]) >= target_quotas.get(cat, 0)
                for cat in target_quotas
            ):
                break

        flat_accepted = [c for cat_list in accepted_by_cat.values() for c in cat_list]
        dev_cases, held_out_cases = stratified_split(flat_accepted, seed=resolved_seed)

        def _count_by_cat(items: list[EvalCase]) -> dict[str, int]:
            counts: dict[str, int] = {}
            for item in items:
                counts[item.category] = counts.get(item.category, 0) + 1
            return counts

        rej_counts: dict[str, int] = {}
        for rej in all_rejected:
            key = f"{rej.gate}: {rej.reason}"
            rej_counts[key] = rej_counts.get(key, 0) + 1

        eval_writer_m = self._eval_model()
        agent_m = definition.model.name
        eval_set = EvalSet(
            agent_name=definition.meta.name,
            eval_writer_model=eval_writer_m,
            agent_model=agent_m,
            same_model_as_agent=(eval_writer_m.strip().lower() == agent_m.strip().lower()),
            same_family_as_agent=(model_family(eval_writer_m) == model_family(agent_m)),
            seed=resolved_seed,
            dev_cases=dev_cases,
            held_out_cases=held_out_cases,
            rejected_cases=all_rejected,
            category_counts=_count_by_cat([*dev_cases, *held_out_cases]),
            dev_category_counts=_count_by_cat(dev_cases),
            held_out_category_counts=_count_by_cat(held_out_cases),
            rejected_counts_by_reason=rej_counts,
            scaled_quotas_note=scaled_note,
        )

        if out_dir is not None:
            eval_set.save(out_dir)

        return eval_set
