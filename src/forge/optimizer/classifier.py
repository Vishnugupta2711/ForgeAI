"""Deterministic-first failure classifier with LLM tie-breaking."""

from __future__ import annotations

from pydantic import BaseModel

from forge.config import load_prompt
from forge.evals.models import EvalCase
from forge.ingest.models import Chunk
from forge.judge.models import EvalCaseResult
from forge.llm.client import LLMClient
from forge.optimizer.models import ClassifiedFailure, FailureCategory


class _TieBreakOutput(BaseModel):
    failure_category: FailureCategory
    subtype: str = ""
    evidence: str = ""


def _has_retrieval_hit_for_case(
    case: EvalCase,
    result: EvalCaseResult,
    chunk_map: dict[str, Chunk] | None = None,
) -> tuple[bool, str]:
    """Check whether expected gold source chunks/docs appear in retrieved chunks."""
    if case.expected_behavior == "refuse" or not case.expected_sources:
        return True, "no expected sources required"

    retrieved_cids = {rc.chunk_id for rc in result.answer_record.retrieved_chunks}
    retrieved_dids = {rc.doc_id for rc in result.answer_record.retrieved_chunks}

    gold_cids = {ref.chunk_id for ref in case.expected_sources}
    gold_dids = {ref.doc_id for ref in case.expected_sources}

    if case.category == "multi_doc":
        if len(gold_dids) >= 2:
            missing_docs = sorted(gold_dids - retrieved_dids)
            if missing_docs:
                return False, f"Multi-doc retrieval missed source doc(s): {missing_docs}"
            return True, "All expected multi-doc sources retrieved."
        matched_chunks = gold_cids & retrieved_cids
        if len(matched_chunks) < 2:
            return False, f"Multi-doc retrieval only hit {len(matched_chunks)}/2 gold chunks."
        return True, "Gold chunks retrieved."

    # Single-doc / edge / tool_use: match exact chunk_id first, or doc_id + key_fact overlap
    if gold_cids & retrieved_cids:
        return True, "Exact gold chunk_id retrieved."

    if chunk_map and (gold_dids & retrieved_dids):
        # If chunk_size changed, check if retrieved chunks from gold doc contain key_facts
        ret_text = "\n".join(
            chunk_map[cid].text for cid in retrieved_cids if cid in chunk_map
        ).lower()
        if case.key_facts and any(kf.lower() in ret_text for kf in case.key_facts):
            return True, "Gold document chunk containing key fact was retrieved."

    return False, (
        f"Gold source(s) {sorted(gold_cids)} not present in retrieved top-k "
        f"{sorted(retrieved_cids)}."
    )


def classify_failure(
    case: EvalCase,
    result: EvalCaseResult,
    *,
    llm_client: LLMClient | None = None,
    chunk_map: dict[str, Chunk] | None = None,
) -> ClassifiedFailure:
    """Classify a failed EvalCaseResult into exactly one primary failure category."""
    v = result.verdict
    out = result.answer_record.output

    # 1. Wrong refusal / missed refusal (authoritative deterministic check)
    if not v.refusal_ok:
        if case.expected_behavior == "refuse" and not out.refused:
            return ClassifiedFailure(
                case_id=case.id,
                eval_category=case.category,
                failure_category="wrong_refusal",
                subtype="missed_refusal",
                evidence="Agent answered an out-of-scope question instead of refusing.",
                question_preview=case.question[:100],
                reasons=v.reasons,
            )
        return ClassifiedFailure(
            case_id=case.id,
            eval_category=case.category,
            failure_category="wrong_refusal",
            subtype="false_refusal",
            evidence=(
                f"Agent refused an answerable question (expected '{case.expected_behavior}')."
            ),
            question_preview=case.question[:100],
            reasons=v.reasons,
        )

    # 2. Retrieval miss (gold chunk/doc absent from top-k)
    retrieved_ok, ret_msg = _has_retrieval_hit_for_case(case, result, chunk_map=chunk_map)
    if not retrieved_ok:
        return ClassifiedFailure(
            case_id=case.id,
            eval_category=case.category,
            failure_category="retrieval_miss",
            subtype="missing_gold_chunk",
            evidence=ret_msg,
            question_preview=case.question[:100],
            reasons=v.reasons,
        )

    # 3. Deterministic tie check: if judge was uncertain or mixed signals exist, allow LLM tie-break
    hallucination_signals = (
        (not v.grounded)
        or (not v.citations_valid)
        or bool(v.ungrounded_numbers)
        or any(not cr.supported for cr in v.claim_results)
    )
    prompt_signals = not v.correct

    if v.uncertain and hallucination_signals and prompt_signals and llm_client is not None:
        prompt = load_prompt("optimizer_classify_tie.v1.md").format(
            case_id=case.id,
            question=case.question,
            expected_behavior=case.expected_behavior,
            grounded=v.grounded,
            correct=v.correct,
            refusal_ok=v.refusal_ok,
            key_fact_recall=v.key_fact_recall,
            reasons="; ".join(v.reasons),
            retrieved_chunks=[rc.chunk_id for rc in result.answer_record.retrieved_chunks],
            expected_sources=[f"{s.doc_id}:{s.chunk_id}" for s in case.expected_sources],
        )
        try:
            tie = llm_client.json(
                prompt,
                schema=_TieBreakOutput,
                role="judge",
                temperature=0.0,
                max_repairs=1,
            )
            return ClassifiedFailure(
                case_id=case.id,
                eval_category=case.category,
                failure_category=tie.failure_category,
                subtype=tie.subtype or "llm_tie_break",
                evidence=tie.evidence or "Resolved via LLM tie-breaker.",
                question_preview=case.question[:100],
                reasons=v.reasons,
                used_llm_tiebreaker=True,
            )
        except Exception:
            pass

    # 4. Hallucination (ungrounded claims, invalid citation, or ungrounded numbers)
    if hallucination_signals:
        ev_parts: list[str] = []
        if not v.citations_valid:
            ev_parts.append("invalid or missing citation quote")
        if v.ungrounded_numbers:
            ev_parts.append(f"ungrounded numbers {v.ungrounded_numbers}")
        if any(not cr.supported for cr in v.claim_results):
            ev_parts.append("unsupported claim(s) in answer")
        return ClassifiedFailure(
            case_id=case.id,
            eval_category=case.category,
            failure_category="hallucination",
            subtype="ungrounded_content",
            evidence="; ".join(ev_parts) or "Answer failed groundedness checks.",
            question_preview=case.question[:100],
            reasons=v.reasons,
        )

    # 5. Prompt defect (context retrieved, refusal ok, grounded ok, but wrong/missed facts)
    return ClassifiedFailure(
        case_id=case.id,
        eval_category=case.category,
        failure_category="prompt_defect",
        subtype="context_ignored_or_incomplete",
        evidence=(
            f"Right context retrieved, but key_fact_recall={v.key_fact_recall:.2f} "
            f"and correctness={v.correct}."
        ),
        question_preview=case.question[:100],
        reasons=v.reasons,
    )


def classify_failures(
    cases: list[EvalCase],
    results: list[EvalCaseResult],
    *,
    llm_client: LLMClient | None = None,
    chunk_map: dict[str, Chunk] | None = None,
) -> list[ClassifiedFailure]:
    """Classify every failed EvalCaseResult in the run."""
    case_by_id = {c.id: c for c in cases}
    failures: list[ClassifiedFailure] = []
    for res in results:
        if res.verdict.passed:
            continue
        case = case_by_id.get(res.case_id)
        if case is None:
            continue
        failures.append(
            classify_failure(case, res, llm_client=llm_client, chunk_map=chunk_map)
        )
    return failures
