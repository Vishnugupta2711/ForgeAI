"""Bounded critic-optimizer loop with rollback, diff/leak guards, and held-out isolation."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from forge.agent.runtime import AgentRuntime
from forge.agent.schema import AgentDefinition, AgentOutput
from forge.config import ForgeConfig, load_prompt
from forge.evals.models import EvalCase, EvalSet
from forge.ingest import ingest_folder
from forge.ingest.models import Chunk, Document
from forge.judge.models import EvalScorecard
from forge.judge.scorer import Judge
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.optimizer.classifier import classify_failures
from forge.optimizer.guards import filter_leaking_patches, verify_diff_guard
from forge.optimizer.models import (
    ClassifiedFailure,
    IterationRecord,
    OptimizationScorecard,
    OptimizerPatch,
    PromptPatch,
    RetrievalPatch,
)
from forge.optimizer.patches import apply_patches, validate_patch_operator_mapping
from forge.retrieval.hybrid import HybridIndex, HybridRetriever
from forge.retrieval.models import RetrievalConfig
from forge.tools import ToolContext


class _ProposedPatchList(BaseModel):
    patches: list[OptimizerPatch] = Field(default_factory=list)


def _category_scores(scorecard: EvalScorecard) -> dict[str, float]:
    return {m.category: m.pass_rate for m in scorecard.category_metrics}


def _patch_signature(patch: OptimizerPatch) -> str:
    if isinstance(patch, RetrievalPatch):
        fields = ",".join(sorted(patch.target_fields.keys()))
        return f"retrieval:{fields}"
    return f"prompt:{patch.section}:{patch.op}"


def _build_index_for_definition(
    docs_dir: Path,
    definition: AgentDefinition,
    llm_client: LLMClient,
    *,
    bm25_only: bool,
) -> tuple[list[Document], list[Chunk], HybridIndex]:
    ingest_res, chunks, _ = ingest_folder(
        docs_dir,
        chunk_size=definition.retrieval.chunk_size,
        overlap=definition.retrieval.overlap,
    )
    if bm25_only:
        return ingest_res.documents, chunks, HybridIndex.build(chunks, llm_client=None)
    try:
        idx = HybridIndex.build(
            chunks,
            llm_client=llm_client,
            embed_model=definition.model.embed_model,
        )
    except LLMProviderError:
        idx = HybridIndex.build(chunks, llm_client=None)
    return ingest_res.documents, chunks, idx


def _render_iteration_log(record: IterationRecord, failures_before: list[ClassifiedFailure]) -> str:
    lines = [
        f"# Optimization Iteration {record.iteration}",
        "",
        f"- **Score Before (dev)**: `{record.score_before * 100:.1f}%`",
        f"- **Score After (dev)**: `{record.score_after * 100:.1f}%`",
        f"- **Accepted**: `{record.accepted}` | **Rolled Back**: `{record.rolled_back}`",
        f"- **Decision Reason**: {record.decision_reason}",
        "",
        "## Dominant Failures Before Patch",
    ]
    if record.failure_counts:
        for cat, cnt in sorted(record.failure_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{cat}`: {cnt}")
    else:
        lines.append("- None")

    lines.extend(["", "## Proposed Patches"])
    if record.proposed_patches:
        for p in record.proposed_patches:
            if isinstance(p, RetrievalPatch):
                lines.append(
                    f"- **RetrievalPatch** `{p.target_fields}` "
                    f"(addresses: {p.addressed_case_ids}): {p.rationale}"
                )
            else:
                lines.append(
                    f"- **PromptPatch** `{p.section}.{p.op}` "
                    f"(addresses: {p.addressed_case_ids}): {p.rationale} | text=`{p.text}`"
                )
    else:
        lines.append("- None")

    if record.rejected_by_leak_guard:
        lines.extend(["", "## Leak Guard Rejections"])
        for msg in record.rejected_by_leak_guard:
            lines.append(f"- {msg}")

    if record.rejected_precheck:
        lines.extend(["", "## Precheck Rejections"])
        for msg in record.rejected_precheck:
            lines.append(f"- {msg}")

    lines.extend(["", "## Failure Case Details"])
    for f in failures_before:
        lines.append(
            f"- **{f.case_id}** (`{f.failure_category}`/{f.subtype}): {f.evidence}"
        )

    lines.append("")
    return "\n".join(lines)


class OptimizerLoop:
    """Run targeted patch optimization over dev evals with strict guards and rollback."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        config: ForgeConfig | None = None,
        max_iterations: int = 3,
        category_regression_tolerance: float = 0.15,
        bm25_only: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or llm_client.config
        self.max_iterations = min(3, max(1, max_iterations))
        self.category_regression_tolerance = category_regression_tolerance
        self.bm25_only = bm25_only

    def _prepare_candidate(
        self,
        base_def: AgentDefinition,
        patches_to_apply: list[OptimizerPatch],
        docs_path: Path,
        documents: list[Document],
        best_index: HybridIndex,
        best_chunk_map: dict[str, Chunk],
    ) -> tuple[AgentDefinition, list[Document], HybridIndex, dict[str, Chunk]]:
        c_def = apply_patches(base_def, patches_to_apply)
        verify_diff_guard(base_def, c_def, patches_to_apply)
        if any(
            isinstance(p, RetrievalPatch) and p.triggers_reindex
            for p in patches_to_apply
        ):
            c_docs, c_chunks, c_idx = _build_index_for_definition(
                docs_path, c_def, self.llm_client, bm25_only=self.bm25_only
            )
            c_map = {c.chunk_id: c for c in c_chunks}
            return c_def, c_docs, c_idx, c_map
        return c_def, documents, best_index, best_chunk_map

    def _eval_split(
        self,
        definition: AgentDefinition,
        cases: list[EvalCase],
        index: HybridIndex,
        documents: list[Document],
        chunk_map: dict[str, Chunk],
        *,
        split: str,
        seed: int,
    ) -> EvalScorecard:
        tool_ctx = ToolContext.from_documents(documents, offline_only=True)
        runtime = AgentRuntime(
            definition,
            index,
            llm_client=self.llm_client,
            tool_context=tool_ctx,
            config=self.config,
        )
        records = [runtime.answer(c.question) for c in cases]
        judge = Judge(self.llm_client, config=self.config, seed=seed)
        return judge.evaluate_suite(
            definition,
            cases,
            records,
            chunk_map,
            split=split,
        )

    def _fallback_patches(
        self,
        failures: list[ClassifiedFailure],
        definition: AgentDefinition,
        rejected_sigs: set[str],
    ) -> list[OptimizerPatch]:
        counts = Counter(f.failure_category for f in failures)
        patches: list[OptimizerPatch] = []
        for cat, _ in counts.most_common():
            if len(patches) >= 2:
                break
            case_ids = [f.case_id for f in failures if f.failure_category == cat][:4]
            candidate: OptimizerPatch | None = None
            if cat == "retrieval_miss":
                if "retrieval:top_k" not in rejected_sigs and definition.retrieval.top_k < 10:
                    candidate = RetrievalPatch(
                        target_fields={"top_k": min(10, definition.retrieval.top_k + 2)},
                        addressed_case_ids=case_ids,
                        rationale="Expand top_k to recover missed source chunks.",
                    )
                elif "retrieval:rerank" not in rejected_sigs and not definition.retrieval.rerank:
                    candidate = RetrievalPatch(
                        target_fields={"rerank": True},
                        addressed_case_ids=case_ids,
                        rationale="Enable lexical reranking to surface relevant chunks.",
                    )
                elif (
                    "retrieval:query_rewrite" not in rejected_sigs
                    and not definition.retrieval.query_rewrite
                ):
                    candidate = RetrievalPatch(
                        target_fields={"query_rewrite": True},
                        addressed_case_ids=case_ids,
                        rationale="Enable query rewriting for retrieval recall.",
                    )
            elif cat == "hallucination":
                if "prompt:grounding_rules:add_rule" not in rejected_sigs:
                    candidate = PromptPatch(
                        section="grounding_rules",
                        op="add_rule",
                        text=(
                            "Never introduce numbers, identifiers, or claims absent from "
                            "the retrieved chunks or tool outputs."
                        ),
                        addressed_case_ids=case_ids,
                        rationale="Tighten grounding against unverified claims and numbers.",
                    )
            elif cat == "wrong_refusal":
                if "prompt:refusal_policy:add_rule" not in rejected_sigs:
                    candidate = PromptPatch(
                        section="refusal_policy",
                        op="add_rule",
                        text=(
                            "Refuse only when no retrieved chunk or attached tool supports "
                            "answering; otherwise answer concisely with citations."
                        ),
                        addressed_case_ids=case_ids,
                        rationale="Calibrate refusal boundary between answerable and OOS asks.",
                    )
            elif cat == "prompt_defect":
                if "prompt:format_rules:add_rule" not in rejected_sigs:
                    candidate = PromptPatch(
                        section="format_rules",
                        op="add_rule",
                        text=(
                            "Include all relevant facts and exact units directly from the "
                            "retrieved passages inside the final answer."
                        ),
                        addressed_case_ids=case_ids,
                        rationale="Ensure retrieved facts are reflected completely in answers.",
                    )
                elif "prompt:extra_rules:add_rule" not in rejected_sigs:
                    candidate = PromptPatch(
                        section="extra_rules",
                        op="add_rule",
                        text=(
                            "Synthesize every required numeric and policy fact explicitly "
                            "before finalizing the JSON response."
                        ),
                        addressed_case_ids=case_ids,
                        rationale="Alternative prompt operator when format_rules is exhausted.",
                    )

            if candidate is not None and _patch_signature(candidate) not in rejected_sigs:
                patches.append(candidate)
        return patches

    def _propose_patches(
        self,
        failures: list[ClassifiedFailure],
        definition: AgentDefinition,
        rejected_sigs: set[str],
    ) -> list[OptimizerPatch]:
        counts = Counter(f.failure_category for f in failures)
        summary_lines = [
            f"Counts by category: {dict(counts)}",
            "Top failure cases:",
        ]
        for f in failures[:12]:
            summary_lines.append(
                f"- {f.case_id} ({f.failure_category}/{f.subtype}): {f.evidence}"
            )

        score_scale = HybridRetriever(
            HybridIndex([]),
            config=RetrievalConfig(
                alpha=definition.retrieval.hybrid_alpha,
                top_k=definition.retrieval.top_k,
                min_score=definition.retrieval.min_score,
            ),
        ).score_scale()

        prompt = load_prompt("optimizer_propose_patches.v1.md").format(
            score_scale_json=json.dumps(score_scale, indent=2),
            retrieval_json=json.dumps(definition.retrieval.model_dump(), indent=2),
            system_prompt_json=json.dumps(definition.system_prompt.model_dump(), indent=2),
            failure_summary="\n".join(summary_lines),
            rejected_ops=sorted(rejected_sigs) or ["(none)"],
        )

        proposed: list[OptimizerPatch] = []
        try:
            res = self.llm_client.json(
                prompt,
                schema=_ProposedPatchList,
                role="judge",
                temperature=0.0,
                max_repairs=2,
            )
            raw_candidates = [
                p for p in res.patches[:2] if _patch_signature(p) not in rejected_sigs
            ]
            proposed, mapping_rejections = validate_patch_operator_mapping(
                raw_candidates, failures, score_scale=score_scale
            )
            if mapping_rejections:
                for cand in raw_candidates:
                    if cand not in proposed:
                        rejected_sigs.add(_patch_signature(cand))
        except Exception:
            proposed = []

        if not proposed:
            raw_fallback = self._fallback_patches(failures, definition, rejected_sigs)
            proposed, _ = validate_patch_operator_mapping(
                raw_fallback, failures, score_scale=score_scale
            )
        return proposed[:2]

    def _precheck_candidate(
        self,
        candidate_def: AgentDefinition,
        patches: list[OptimizerPatch],
        dev_cases: list[EvalCase],
        cand_index: HybridIndex,
        cand_docs: list[Document],
    ) -> tuple[bool, str | None]:
        """Cheap pre-eval sample (3-5 dev questions, 0 judge calls) blocking bad patches."""
        answerable = [c for c in dev_cases if c.expected_behavior != "refuse"]
        if not answerable:
            return True, None

        addressed_ids: list[str] = []
        for p in patches:
            addressed_ids.extend(p.addressed_case_ids)
        addressed_set = set(addressed_ids)
        prioritized = [c for c in answerable if c.id in addressed_set] + [
            c for c in answerable if c.id not in addressed_set
        ]
        sample_size = min(5, len(prioritized))
        if sample_size > 3:
            sample = prioritized[:sample_size]
        else:
            sample = prioritized[: min(len(prioritized), 3)]

        # 1. Retrieval-only check (no LLM judge or agent chat calls)
        use_dense = (
            (not self.bm25_only)
            and cand_index.embeddings is not None
            and candidate_def.retrieval.hybrid_alpha > 0.0
        )
        retriever = HybridRetriever(
            cand_index,
            llm_client=self.llm_client if use_dense else None,
            config=RetrievalConfig(
                alpha=candidate_def.retrieval.hybrid_alpha if use_dense else 0.0,
                top_k=candidate_def.retrieval.top_k,
                min_score=candidate_def.retrieval.min_score,
                query_rewrite=False,
            ),
        )
        empty_hits = sum(1 for c in sample if not retriever.search(c.question))
        if empty_hits > (len(sample) / 2.0):
            return False, (
                f"rejected_precheck: retrieval emptied for {empty_hits}/{len(sample)} "
                f"sampled dev questions (min_score={candidate_def.retrieval.min_score})."
            )

        # 2. Structural schema validity + cheap agent-only schema probe for PromptPatch
        try:
            AgentOutput(
                answer="ok",
                citations=[],
                refused=True,
                refusal_reason="probe",
                confidence=0.5,
            )
        except Exception as exc:
            return False, f"rejected_precheck: output contract schema invalid ({exc})."

        if any(isinstance(p, PromptPatch) for p in patches):
            probe = sample[: min(3, len(sample))]
            tool_ctx = ToolContext.from_documents(cand_docs, offline_only=True)
            runtime = AgentRuntime(
                candidate_def,
                cand_index,
                llm_client=self.llm_client,
                tool_context=tool_ctx,
                config=self.config,
            )
            schema_fails = sum(1 for c in probe if runtime.answer(c.question).contract_failed)
            if schema_fails > (len(probe) / 2.0):
                return False, (
                    f"rejected_precheck: contract schema validation failed on "
                    f"{schema_fails}/{len(probe)} sampled dev questions."
                )

        return True, None

    def run(
        self,
        definition: AgentDefinition,
        eval_set: EvalSet,
        docs_dir: Path | str,
        *,
        out_dir: Path | str,
        deadline_monotonic: float | None = None,
    ) -> tuple[AgentDefinition, OptimizationScorecard]:
        """Optimize definition on dev_cases only; score held_out at baseline and end."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        docs_path = Path(docs_dir)

        best_def = definition.model_copy(deep=True)
        documents, chunks, best_index = _build_index_for_definition(
            docs_path, best_def, self.llm_client, bm25_only=self.bm25_only
        )
        best_chunk_map = {c.chunk_id: c for c in chunks}

        # Baseline evaluation on dev AND held_out (once at start)
        best_dev_scorecard = self._eval_split(
            best_def,
            eval_set.dev_cases,
            best_index,
            documents,
            best_chunk_map,
            split="dev",
            seed=eval_set.seed,
        )
        baseline_dev_score = best_dev_scorecard.overall_pass_rate
        baseline_held_scorecard = (
            self._eval_split(
                best_def,
                eval_set.held_out_cases,
                best_index,
                documents,
                best_chunk_map,
                split="held_out",
                seed=eval_set.seed,
            )
            if eval_set.held_out_cases
            else None
        )
        baseline_held_score = (
            baseline_held_scorecard.overall_pass_rate if baseline_held_scorecard else 0.0
        )

        current_dev_scorecard = best_dev_scorecard
        rejected_sigs: set[str] = set()
        consecutive_no_improve = 0
        iterations: list[IterationRecord] = []
        stop_reason = f"max_iterations ({self.max_iterations}) reached"

        for iter_idx in range(1, self.max_iterations + 1):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                stop_reason = "time_budget_exceeded"
                break
            failures = classify_failures(
                eval_set.dev_cases,
                current_dev_scorecard.results,
                llm_client=self.llm_client,
                chunk_map=best_chunk_map,
            )
            if not failures:
                stop_reason = "all_dev_cases_passed"
                break

            iter_dir = out_path / "iterations" / f"iter_{iter_idx}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            failure_counts = dict(Counter(f.failure_category for f in failures))
            score_before = best_dev_scorecard.overall_pass_rate
            cats_before = _category_scores(best_dev_scorecard)

            raw_patches = self._propose_patches(failures, best_def, rejected_sigs)
            clean_patches, leak_rejections = filter_leaking_patches(
                raw_patches, eval_set.dev_cases, eval_set.held_out_cases
            )

            if not clean_patches:
                rec = IterationRecord(
                    iteration=iter_idx,
                    score_before=score_before,
                    score_after=score_before,
                    category_scores_before=cats_before,
                    category_scores_after=cats_before,
                    failure_counts=failure_counts,
                    proposed_patches=raw_patches,
                    rejected_by_leak_guard=leak_rejections,
                    accepted=False,
                    rolled_back=True,
                    decision_reason=(
                        "No valid non-leaking patches proposed."
                        if leak_rejections or not raw_patches
                        else "Empty patch set."
                    ),
                )
                iterations.append(rec)
                best_def.save_yaml(iter_dir / "definition.yaml")
                (iter_dir / "results.json").write_text(
                    current_dev_scorecard.model_dump_json(indent=2), encoding="utf-8"
                )
                (iter_dir / "failures.json").write_text(
                    json.dumps([f.model_dump() for f in failures], indent=2), encoding="utf-8"
                )
                (iter_dir / "patch.json").write_text(
                    rec.model_dump_json(indent=2), encoding="utf-8"
                )
                (iter_dir / "log.md").write_text(
                    _render_iteration_log(rec, failures), encoding="utf-8"
                )
                consecutive_no_improve += 1
                if consecutive_no_improve >= 2:
                    stop_reason = "early_stop_two_consecutive_no_improvement"
                    break
                continue

            candidate_def, cand_docs, cand_index, cand_chunk_map = self._prepare_candidate(
                best_def, clean_patches, docs_path, documents, best_index, best_chunk_map
            )
            precheck_rejections: list[str] = []
            pre_ok, pre_reason = self._precheck_candidate(
                candidate_def, clean_patches, eval_set.dev_cases, cand_index, cand_docs
            )
            if not pre_ok:
                precheck_rejections.append(pre_reason or "rejected_precheck")
                for p in clean_patches:
                    rejected_sigs.add(_patch_signature(p))

                # Try one alternative operator in this iteration (skip full eval on bad patch)
                alt_scale = HybridRetriever(
                    HybridIndex([]),
                    config=RetrievalConfig(
                        alpha=best_def.retrieval.hybrid_alpha,
                        top_k=best_def.retrieval.top_k,
                        min_score=best_def.retrieval.min_score,
                    ),
                ).score_scale()
                alt_raw = self._fallback_patches(failures, best_def, rejected_sigs)
                alt_mapped, _ = validate_patch_operator_mapping(
                    alt_raw, failures, score_scale=alt_scale
                )
                alt_clean, alt_leaks = filter_leaking_patches(
                    alt_mapped, eval_set.dev_cases, eval_set.held_out_cases
                )
                leak_rejections.extend(alt_leaks)
                alt_ready = False
                if alt_clean:
                    candidate_def, cand_docs, cand_index, cand_chunk_map = (
                        self._prepare_candidate(
                            best_def, alt_clean, docs_path, documents, best_index, best_chunk_map
                        )
                    )
                    alt_ok, alt_reason = self._precheck_candidate(
                        candidate_def, alt_clean, eval_set.dev_cases, cand_index, cand_docs
                    )
                    if alt_ok:
                        clean_patches = alt_clean
                        alt_ready = True
                    else:
                        precheck_rejections.append(alt_reason or "rejected_precheck")
                        for p in alt_clean:
                            rejected_sigs.add(_patch_signature(p))

                if not alt_ready:
                    rec = IterationRecord(
                        iteration=iter_idx,
                        score_before=score_before,
                        score_after=score_before,
                        category_scores_before=cats_before,
                        category_scores_after=cats_before,
                        failure_counts=failure_counts,
                        proposed_patches=clean_patches,
                        rejected_by_leak_guard=leak_rejections,
                        rejected_precheck=precheck_rejections,
                        accepted=False,
                        rolled_back=True,
                        decision_reason="; ".join(precheck_rejections),
                    )
                    iterations.append(rec)
                    best_def.save_yaml(iter_dir / "definition.yaml")
                    (iter_dir / "results.json").write_text(
                        current_dev_scorecard.model_dump_json(indent=2), encoding="utf-8"
                    )
                    (iter_dir / "failures.json").write_text(
                        json.dumps([f.model_dump() for f in failures], indent=2),
                        encoding="utf-8",
                    )
                    (iter_dir / "patch.json").write_text(
                        rec.model_dump_json(indent=2), encoding="utf-8"
                    )
                    (iter_dir / "log.md").write_text(
                        _render_iteration_log(rec, failures), encoding="utf-8"
                    )
                    consecutive_no_improve += 1
                    if consecutive_no_improve >= 2:
                        stop_reason = "early_stop_two_consecutive_no_improvement"
                        break
                    continue

            cand_scorecard = self._eval_split(
                candidate_def,
                eval_set.dev_cases,
                cand_index,
                cand_docs,
                cand_chunk_map,
                split="dev",
                seed=eval_set.seed,
            )
            score_after = cand_scorecard.overall_pass_rate
            cats_after = _category_scores(cand_scorecard)

            # Check per-category regression tolerance
            regressed_cats: list[str] = []
            for cat, prev_rate in cats_before.items():
                new_rate = cats_after.get(cat, prev_rate)
                if (prev_rate - new_rate) > self.category_regression_tolerance + 1e-9:
                    regressed_cats.append(f"{cat} ({prev_rate:.2f} -> {new_rate:.2f})")

            improved = score_after > score_before + 1e-9
            accepted = improved and not regressed_cats
            rolled_back = not accepted

            if accepted:
                decision_reason = (
                    f"Accepted: dev score improved {score_before * 100:.1f}% -> "
                    f"{score_after * 100:.1f}% with no category regression."
                )
                best_def = candidate_def
                best_index = cand_index
                documents = cand_docs
                best_chunk_map = cand_chunk_map
                best_dev_scorecard = cand_scorecard
                current_dev_scorecard = cand_scorecard
                consecutive_no_improve = 0
            else:
                if regressed_cats:
                    decision_reason = (
                        "Rolled back: category regression exceeded tolerance in "
                        + ", ".join(regressed_cats)
                    )
                else:
                    decision_reason = (
                        f"Rolled back: dev score did not improve "
                        f"({score_before * 100:.1f}% -> {score_after * 100:.1f}%)."
                    )
                for p in clean_patches:
                    rejected_sigs.add(_patch_signature(p))
                consecutive_no_improve += 1

            rec = IterationRecord(
                iteration=iter_idx,
                score_before=score_before,
                score_after=score_after,
                category_scores_before=cats_before,
                category_scores_after=cats_after,
                failure_counts=failure_counts,
                proposed_patches=clean_patches,
                rejected_by_leak_guard=leak_rejections,
                rejected_precheck=precheck_rejections,
                accepted=accepted,
                rolled_back=rolled_back,
                decision_reason=decision_reason,
            )
            iterations.append(rec)

            # Persist per-iteration artifacts
            (candidate_def if accepted else best_def).save_yaml(iter_dir / "definition.yaml")
            cand_scorecard.save(iter_dir)
            (iter_dir / "failures.json").write_text(
                json.dumps([f.model_dump() for f in failures], indent=2), encoding="utf-8"
            )
            (iter_dir / "patch.json").write_text(
                rec.model_dump_json(indent=2), encoding="utf-8"
            )
            (iter_dir / "log.md").write_text(
                _render_iteration_log(rec, failures), encoding="utf-8"
            )

            if best_dev_scorecard.overall_pass_rate >= 1.0 - 1e-9:
                stop_reason = "all_dev_cases_passed"
                break
            if consecutive_no_improve >= 2:
                stop_reason = "early_stop_two_consecutive_no_improvement"
                break

        # Held-out evaluation ONCE at the end on best_def
        final_held_scorecard = (
            self._eval_split(
                best_def,
                eval_set.held_out_cases,
                best_index,
                documents,
                best_chunk_map,
                split="held_out",
                seed=eval_set.seed,
            )
            if eval_set.held_out_cases
            else None
        )
        final_held_score = (
            final_held_scorecard.overall_pass_rate if final_held_scorecard else 0.0
        )

        best_def.save_yaml(out_path / "best_definition.yaml")
        best_dev_scorecard.save(out_path / "final_dev_eval")
        if final_held_scorecard is not None:
            final_held_scorecard.save(out_path / "final_held_out_eval")

        opt_scorecard = OptimizationScorecard(
            agent_name=best_def.meta.name,
            baseline_dev_score=baseline_dev_score,
            final_dev_score=best_dev_scorecard.overall_pass_rate,
            baseline_held_out_score=baseline_held_score,
            final_held_out_score=final_held_score,
            baseline_held_out_by_category=(
                _category_scores(baseline_held_scorecard) if baseline_held_scorecard else {}
            ),
            final_held_out_by_category=(
                _category_scores(final_held_scorecard) if final_held_scorecard else {}
            ),
            iterations_completed=len(iterations),
            stop_reason=stop_reason,
            iterations=iterations,
        )
        opt_scorecard.save(out_path / "scorecard.json")
        return best_def, opt_scorecard
