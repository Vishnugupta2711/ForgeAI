"""Canary self-test suite for measuring ForgeAI Judge reliability."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from forge.agent.schema import (
    AgentOutput,
    AnswerRecord,
    Citation,
    RetrievedChunkRecord,
)
from forge.config import ForgeConfig
from forge.evals.gates import verify_key_fact
from forge.evals.models import EvalCase, EvalProvenance, EvalSet, EvalSourceRef
from forge.ingest.models import Chunk
from forge.judge.deterministic import _NUMBER_RE, canonicalize_text, compute_key_fact_recall
from forge.judge.models import (
    CanaryCaseResult,
    CanaryMetric,
    JudgeReliabilityReport,
)
from forge.judge.scorer import Judge
from forge.llm.client import LLMClient

_DIGIT_RE = _NUMBER_RE
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _mutate_numbers(text: str, key_facts: list[str] | None = None) -> str:
    # Prefer mutating a standalone key-fact number so hyphenated IDs (SP-1026) stay untouched
    fact_nums = (
        {m for kf in key_facts for m in _NUMBER_RE.findall(kf)} if key_facts else set()
    )
    matches = list(_NUMBER_RE.finditer(text))
    target = next((m for m in matches if m.group(1) in fact_nums), matches[0] if matches else None)
    if target is not None:
        raw_num = target.group(1).replace(",", "")
        mutated = str(int(float(raw_num)) + 37)
        return f"{text[:target.start()]}{mutated}{text[target.end():]}"
    return f"{text} (total 9876 units)"


def _is_csv_row_chunk(chunk: Chunk) -> bool:
    if chunk.metadata.get("doc_type") == "csv" or "::row-" in chunk.chunk_id:
        return True
    lines = [ln.strip() for ln in chunk.text.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(":" in ln and "_" in ln.split(":", 1)[0] for ln in lines[:2])


def _render_csv_natural_sentence(chunk: Chunk, key_facts: list[str]) -> str:
    """Turn a CSV row chunk + key facts into a natural prose sentence."""
    fields: dict[str, str] = {}
    ordered_keys: list[str] = []
    for ln in chunk.text.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            key_clean = k.strip()
            fields[key_clean.lower()] = v.strip()
            ordered_keys.append(key_clean)

    subject_val = next(iter(fields.values()), chunk.doc_id)
    for id_col in ("part_no", "sensor_id", "id", "name", "description"):
        if id_col in fields and fields[id_col]:
            subject_val = fields[id_col]
            break

    clauses: list[str] = []
    for kf in key_facts:
        if ":" in kf:
            col_part, val_part = kf.split(":", 1)
            col_words = col_part.strip().replace("_", " ")
            clauses.append(f"the {col_words} is {val_part.strip()}")
        else:
            clauses.append(kf.strip().rstrip("."))

    if clauses:
        return f"For {subject_val}, " + ", and ".join(clauses) + "."
    return f"For {subject_val}, recorded catalogue details apply."


def _build_natural_gold_answer(base: EvalCase, gold_chunks: list[Chunk]) -> tuple[str, str]:
    """Build natural sentences covering base.key_facts and return (answer, valid_quote)."""
    sentences: list[str] = []
    seen_lower: set[str] = set()
    first_quote = ""

    # Separate prose chunks from CSV row chunks
    prose_chunks = [c for c in gold_chunks if not _is_csv_row_chunk(c)]
    csv_chunks = [c for c in gold_chunks if _is_csv_row_chunk(c)]

    remaining_facts: list[str] = []
    for kf in base.key_facts:
        matched_sent = ""
        for ch in prose_chunks:
            # Join wrapped PDF lines before sentence extraction
            flat = re.sub(r"\s*\n\s*", " ", ch.text).strip()
            for sent in _SENTENCE_SPLIT_RE.split(flat):
                s_clean = sent.strip()
                if s_clean and verify_key_fact(kf, s_clean, threshold=0.80):
                    matched_sent = s_clean if s_clean.endswith((".", "!", "?")) else f"{s_clean}."
                    if not first_quote:
                        first_quote = s_clean[:120]
                    break
            if matched_sent:
                break
        if matched_sent:
            low = matched_sent.lower()
            if low not in seen_lower:
                seen_lower.add(low)
                sentences.append(matched_sent)
        else:
            remaining_facts.append(kf)

    if remaining_facts and csv_chunks:
        csv_facts = [
            kf
            for kf in remaining_facts
            if any(verify_key_fact(kf, c.text, threshold=0.80) for c in csv_chunks)
        ]
        if csv_facts:
            target_csv = next(
                (c for c in csv_chunks if verify_key_fact(csv_facts[0], c.text, threshold=0.80)),
                csv_chunks[0],
            )
            natural_csv = _render_csv_natural_sentence(target_csv, csv_facts)
            sentences.append(natural_csv)
            if not first_quote:
                first_quote = target_csv.text.splitlines()[0][:120]
            remaining_facts = [kf for kf in remaining_facts if kf not in csv_facts]

    for kf in remaining_facts:
        s = kf.strip()
        if ":" in s and "_" in s.split(":", 1)[0]:
            c_part, v_part = s.split(":", 1)
            s = f"The {c_part.strip().replace('_', ' ')} is {v_part.strip()}."
        elif not s.endswith((".", "!", "?")):
            s = f"The documented policy confirms that {s}."
        if s.lower() not in seen_lower:
            seen_lower.add(s.lower())
            sentences.append(s)

    answer = " ".join(sentences).strip()
    recall, _, missed = compute_key_fact_recall(base.key_facts, answer)
    if recall < 1.0 and missed:
        # Ensure all canonical key-fact tokens/numbers are represented cleanly in natural prose
        answer = f"{answer} In addition, {'; '.join(missed)}.".strip()

    if not first_quote and gold_chunks:
        p0 = gold_chunks[0]
        first_quote = next(
            (
                kf[:120]
                for kf in base.key_facts
                if canonicalize_text(kf) in canonicalize_text(p0.text)
            ),
            p0.text.splitlines()[0][:120],
        )
    return answer, first_quote


class CanarySpec(BaseModel):
    """A single synthetic canary case paired with expected pass/fail outcome."""

    canary_id: str
    canary_type: str
    expected_pass: bool
    eval_case: EvalCase
    answer_record: AnswerRecord


def _fallback_eval_set_from_chunks(chunks: list[Chunk]) -> EvalSet:
    prov = EvalProvenance(
        prompt_file="selftest_synthetic",
        prompt_hash="selftest",
        model="selftest",
        seed=42,
    )
    answer_cases: list[EvalCase] = []
    for idx, ch in enumerate(chunks[:3], start=1):
        line = next((ln.strip() for ln in ch.text.splitlines() if len(ln.strip()) >= 12), ch.text)
        answer_cases.append(
            EvalCase(
                id=f"ST-A{idx}",
                category="single_doc",
                question=f"What does {ch.doc_id} state about {line[:30]}?",
                expected_behavior="answer",
                expected_sources=[EvalSourceRef(doc_id=ch.doc_id, chunk_id=ch.chunk_id)],
                key_facts=[line[:80]],
                provenance=prov,
            )
        )
    refuse_case = EvalCase(
        id="ST-R1",
        category="out_of_scope",
        question="What is the classified budget projection not mentioned in the documents?",
        expected_behavior="refuse",
        provenance=prov,
    )
    return EvalSet(
        agent_name="selftest-target",
        eval_writer_model="selftest",
        agent_model="selftest",
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[*answer_cases, refuse_case],
        held_out_cases=[],
    )


def build_canary_suite(
    eval_set: EvalSet | None,
    chunk_map: dict[str, Chunk],
    *,
    max_answer_bases: int = 2,
) -> list[CanarySpec]:
    """Construct all 8 canary categories from an EvalSet and corpus chunk_map."""
    all_chunks = list(chunk_map.values())
    if not all_chunks:
        raise ValueError("Cannot build canary suite from an empty chunk_map.")

    source_eval = eval_set if eval_set is not None else _fallback_eval_set_from_chunks(all_chunks)
    all_cases = [*source_eval.dev_cases, *source_eval.held_out_cases]

    answer_bases = [
        c
        for c in all_cases
        if c.expected_behavior == "answer"
        and c.key_facts
        and any(ref.chunk_id in chunk_map for ref in c.expected_sources)
    ]
    if not answer_bases:
        answer_bases = _fallback_eval_set_from_chunks(all_chunks).dev_cases[:2]

    refuse_bases = [c for c in all_cases if c.expected_behavior == "refuse"]
    if not refuse_bases:
        refuse_bases = [
            EvalCase(
                id="ST-R1",
                category="out_of_scope",
                question="Provide confidential external info absent from the corpus.",
                expected_behavior="refuse",
                provenance=answer_bases[0].provenance,
            )
        ]

    canaries: list[CanarySpec] = []
    # Prefer an answer base that contains a digit for clean numeric corruption
    digit_bases = [b for b in answer_bases if any(_DIGIT_RE.search(kf) for kf in b.key_facts)]
    selected_bases = (digit_bases + [b for b in answer_bases if b not in digit_bases])[
        :max_answer_bases
    ]

    for idx, base in enumerate(selected_bases, start=1):
        gold_chunks = [
            chunk_map[ref.chunk_id]
            for ref in base.expected_sources
            if ref.chunk_id in chunk_map
        ]
        if not gold_chunks:
            continue
        primary = gold_chunks[0]
        retrieved = [
            RetrievedChunkRecord(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                score=0.016,
                page=ch.page,
                section_heading=ch.section_heading,
            )
            for ch in gold_chunks
        ]
        gold_answer_text, gold_quote = _build_natural_gold_answer(base, gold_chunks)
        valid_citation = Citation(
            doc_id=primary.doc_id,
            chunk_id=primary.chunk_id,
            quote=gold_quote[:120],
        )

        def _rec(output: AgentOutput, ret=retrieved, q=base.question) -> AnswerRecord:
            return AnswerRecord(
                question=q,
                output=output,
                retrieved_chunks=ret,
                tool_calls=[],
                trace=["canary"],
                effective_query=q,
                model="canary",
            )

        # 1. gold_answer (expected pass)
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-GOLD-{idx}",
                canary_type="gold_answer",
                expected_pass=True,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer=gold_answer_text,
                        citations=[valid_citation],
                        refused=False,
                        confidence=0.95,
                    )
                ),
            )
        )

        # 2. numerically_corrupted (expected fail)
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-NUM-{idx}",
                canary_type="numerically_corrupted",
                expected_pass=False,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer=_mutate_numbers(gold_answer_text, base.key_facts),
                        citations=[valid_citation],
                        refused=False,
                        confidence=0.9,
                    )
                ),
            )
        )

        # 3. hallucinated_claim (expected fail)
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-HALL-{idx}",
                canary_type="hallucinated_claim",
                expected_pass=False,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer=(
                            f"{gold_answer_text} Additionally, an undocumented emergency override "
                            "protocol requires secondary regional audit sign-off "
                            "under bulletin 7741."
                        ),
                        citations=[valid_citation],
                        refused=False,
                        confidence=0.85,
                    )
                ),
            )
        )

        # 4. wrong_document (expected fail)
        other_chunk = next(
            (c for c in all_chunks if c.doc_id != primary.doc_id),
            next((c for c in all_chunks if c.chunk_id != primary.chunk_id), primary),
        )
        other_line = next(
            (ln.strip() for ln in other_chunk.text.splitlines() if len(ln.strip()) >= 8),
            other_chunk.text[:80],
        )
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-WRONGDOC-{idx}",
                canary_type="wrong_document",
                expected_pass=False,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer=other_line,
                        citations=[
                            Citation(
                                doc_id=other_chunk.doc_id,
                                chunk_id=other_chunk.chunk_id,
                                quote=other_line[:100],
                            )
                        ],
                        refused=False,
                        confidence=0.75,
                    ),
                    ret=[
                        RetrievedChunkRecord(
                            chunk_id=other_chunk.chunk_id,
                            doc_id=other_chunk.doc_id,
                            score=0.015,
                        )
                    ],
                ),
            )
        )

        # 5. padded_verbose (expected pass)
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-VERBOSE-{idx}",
                canary_type="padded_verbose",
                expected_pass=True,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer=(
                            "Thank you for asking. After carefully reviewing the relevant "
                            "documentation, here is the verified summary: "
                            f"{gold_answer_text} "
                            "Please consult the cited passage for reference."
                        ),
                        citations=[valid_citation],
                        refused=False,
                        confidence=0.92,
                    )
                ),
            )
        )

        # 6. terse_correct (expected pass)
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-TERSE-{idx}",
                canary_type="terse_correct",
                expected_pass=True,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer=gold_answer_text,
                        citations=[valid_citation],
                        refused=False,
                        confidence=0.9,
                    )
                ),
            )
        )

        # 7. wrongful_refusal (expected fail)
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-WREF-{idx}",
                canary_type="wrongful_refusal",
                expected_pass=False,
                eval_case=base,
                answer_record=_rec(
                    AgentOutput(
                        answer="I cannot answer this question because I lack information.",
                        citations=[],
                        refused=True,
                        refusal_reason="Declined despite available evidence.",
                        confidence=0.8,
                    )
                ),
            )
        )

    # 8. correct_refusal (expected pass)
    for idx, r_case in enumerate(refuse_bases[:max_answer_bases], start=1):
        canaries.append(
            CanarySpec(
                canary_id=f"CAN-CREF-{idx}",
                canary_type="correct_refusal",
                expected_pass=True,
                eval_case=r_case,
                answer_record=AnswerRecord(
                    question=r_case.question,
                    output=AgentOutput(
                        answer="The provided documents do not cover this topic.",
                        citations=[],
                        refused=True,
                        refusal_reason="Out of scope of the grounded documents.",
                        confidence=0.95,
                    ),
                    retrieved_chunks=[],
                    tool_calls=[],
                    trace=["canary"],
                    effective_query=r_case.question,
                    model="canary",
                ),
            )
        )

    return canaries


def run_judge_selftest(
    llm_client: LLMClient,
    chunk_map: dict[str, Chunk],
    *,
    eval_set: EvalSet | None = None,
    config: ForgeConfig | None = None,
    max_answer_bases: int = 2,
    out_path: Path | str | None = None,
) -> JudgeReliabilityReport:
    """Execute the canary suite against Judge and report false-pass / false-fail rates."""
    cfg = config or llm_client.config
    judge = Judge(llm_client, config=cfg)
    canaries = build_canary_suite(eval_set, chunk_map, max_answer_bases=max_answer_bases)

    case_results: list[CanaryCaseResult] = []
    grouped: dict[str, list[CanaryCaseResult]] = {}
    for spec in canaries:
        verdict = judge.evaluate_case(spec.eval_case, spec.answer_record, chunk_map)
        item = CanaryCaseResult(
            canary_id=spec.canary_id,
            canary_type=spec.canary_type,
            expected_pass=spec.expected_pass,
            passed_via_fallback=verdict.passed_via_fallback,
            verdict=verdict,
        )
        case_results.append(item)
        grouped.setdefault(spec.canary_type, []).append(item)

    per_type: list[CanaryMetric] = []
    total_false_passes = 0
    total_false_fails = 0
    total_negatives = 0
    total_positives = 0
    total_clean_passes = 0
    total_fallback_passes = 0

    for c_type in sorted(grouped):
        items = grouped[c_type]
        exp_pass = items[0].expected_pass
        n = len(items)
        act_pass = sum(1 for i in items if i.verdict.passed)
        p_fallback = sum(1 for i in items if i.verdict.passed and i.verdict.passed_via_fallback)
        p_clean = act_pass - p_fallback
        fp = sum(1 for i in items if (not i.expected_pass) and i.verdict.passed)
        ff = sum(1 for i in items if i.expected_pass and (not i.verdict.passed))
        if exp_pass:
            total_positives += n
            total_false_fails += ff
            total_clean_passes += p_clean
            total_fallback_passes += p_fallback
        else:
            total_negatives += n
            total_false_passes += fp

        per_type.append(
            CanaryMetric(
                canary_type=c_type,
                expected_pass=exp_pass,
                total=n,
                actual_pass=act_pass,
                passed_cleanly=p_clean,
                passed_via_fallback=p_fallback,
                false_passes=fp,
                false_fails=ff,
                false_pass_rate=round(fp / n, 4) if (not exp_pass and n) else 0.0,
                false_fail_rate=round(ff / n, 4) if (exp_pass and n) else 0.0,
            )
        )

    report = JudgeReliabilityReport(
        judge_model=cfg.models.judge,
        total_canaries=len(case_results),
        overall_false_pass_rate=(
            round(total_false_passes / total_negatives, 4) if total_negatives else 0.0
        ),
        overall_false_fail_rate=(
            round(total_false_fails / total_positives, 4) if total_positives else 0.0
        ),
        overall_clean_pass_rate=(
            round(total_clean_passes / total_positives, 4) if total_positives else 0.0
        ),
        passed_cleanly_count=total_clean_passes,
        passed_via_fallback_count=total_fallback_passes,
        per_canary_type=per_type,
        cases=case_results,
    )

    if out_path is not None:
        report.save(out_path)

    return report
