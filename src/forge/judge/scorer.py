"""Two-layer ForgeAI Judge with deterministic checks, claim verification, and bias defences."""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from forge.agent.schema import AgentDefinition, AnswerRecord
from forge.config import ForgeConfig, load_prompt
from forge.evals.generator import model_family
from forge.evals.models import EvalCase
from forge.ingest.models import Chunk
from forge.judge.deterministic import (
    check_citation_validity,
    check_refusal,
    compute_key_fact_recall,
    find_ungrounded_numbers,
    pearson_correlation,
    verify_quote_in_evidence,
    wilson_interval,
)
from forge.judge.models import (
    CaseVerdict,
    CategoryMetric,
    ClaimVerification,
    EvalCaseResult,
    EvalScorecard,
)
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError


class _ExtractedClaims(BaseModel):
    claims: list[str] = Field(default_factory=list)


class _ClaimVerdictItem(BaseModel):
    claim_index: int
    supported: bool
    quote: str = ""


class _ClaimVerdictList(BaseModel):
    verdicts: list[_ClaimVerdictItem] = Field(default_factory=list)


class _CorrectnessVerdict(BaseModel):
    correct: bool
    reason: str = ""


class Judge:
    """Automated judge combining deterministic checks with bias-defended LLM verification."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        config: ForgeConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or llm_client.config
        self.seed = self.config.generation.seed if seed is None else seed

    def _judge_model(self) -> str:
        return self.config.models.judge

    def _verify_claims_pass(
        self,
        ordered_claims: list[str],
        evidence_block: str,
        evidence_texts: list[str],
    ) -> dict[str, ClaimVerification]:
        if not ordered_claims:
            return {}
        claims_block = "\n".join(
            f"{idx}. {claim}" for idx, claim in enumerate(ordered_claims, start=1)
        )
        prompt = load_prompt("judge_verify_claims.v1.md").format(
            claims_block=claims_block,
            evidence_block=evidence_block,
        )
        parsed = self.llm_client.json(
            prompt,
            schema=_ClaimVerdictList,
            model=self._judge_model(),
            temperature=0.0,
            seed=self.seed,
            role="judge",
            max_repairs=2,
        )
        by_index = {v.claim_index: v for v in parsed.verdicts}
        result: dict[str, ClaimVerification] = {}
        for idx, claim_text in enumerate(ordered_claims, start=1):
            item = by_index.get(idx)
            if item is None:
                result[claim_text] = ClaimVerification(
                    claim=claim_text,
                    supported=False,
                    quote="",
                    verified_substring=False,
                )
                continue
            quote_ok = (
                verify_quote_in_evidence(item.quote, evidence_texts)
                if item.supported
                else False
            )
            result[claim_text] = ClaimVerification(
                claim=claim_text,
                supported=bool(item.supported and quote_ok),
                quote=item.quote,
                verified_substring=quote_ok,
            )
        return result

    def _judge_claim_groundedness(
        self,
        answer_text: str,
        evidence_block: str,
        evidence_texts: list[str],
        *,
        deterministic_ok: bool,
    ) -> tuple[bool, list[ClaimVerification], bool, list[str]]:
        """Extract claims and run 2-pass order-shuffled verification for position bias."""
        notes: list[str] = []
        extract_prompt = load_prompt("judge_extract_claims.v1.md").format(answer=answer_text)
        try:
            extracted = self.llm_client.json(
                extract_prompt,
                schema=_ExtractedClaims,
                model=self._judge_model(),
                temperature=0.0,
                seed=self.seed,
                role="judge",
                max_repairs=2,
            )
            claims = [c.strip() for c in extracted.claims if c.strip()]
        except LLMProviderError:
            raise
        except Exception as exc:
            notes.append(f"Claim extraction failed ({exc}); using deterministic groundedness.")
            return deterministic_ok, [], True, notes

        if not claims:
            return deterministic_ok, [], False, notes

        # Pass 1: forward order
        try:
            pass1 = self._verify_claims_pass(claims, evidence_block, evidence_texts)
        except LLMProviderError:
            raise
        except Exception as exc:
            notes.append(f"Claim verification pass 1 failed ({exc}); deterministic fallback.")
            return deterministic_ok, [], True, notes

        # Pass 2: reversed / shuffled order to detect position bias
        if len(claims) >= 2:
            shuffled = list(reversed(claims))
            if len(claims) > 2:
                rng = random.Random(self.seed)
                rng.shuffle(shuffled)
                if shuffled == claims:
                    shuffled = list(reversed(claims))
            try:
                pass2 = self._verify_claims_pass(shuffled, evidence_block, evidence_texts)
            except LLMProviderError:
                raise
            except Exception as exc:
                notes.append(f"Claim verification pass 2 failed ({exc}); deterministic fallback.")
                return deterministic_ok, list(pass1.values()), True, notes

            disagreed = any(
                pass1[c].supported != pass2[c].supported for c in claims if c in pass2
            )
            if disagreed:
                notes.append(
                    "Position bias disagreement between forward and shuffled claim verification; "
                    "falling back to deterministic groundedness."
                )
                return deterministic_ok, list(pass1.values()), True, notes

        verdicts = [pass1[c] for c in claims]
        unverified_quotes = [v for v in verdicts if not v.supported and v.quote.strip()]
        if unverified_quotes:
            notes.append("Rejected unsupported or fabricated-quote claim(s).")
        all_supported = all(v.supported for v in verdicts)
        return all_supported, verdicts, False, notes

    def _judge_ambiguous_correctness(
        self,
        question: str,
        key_facts: list[str],
        answer_text: str,
        *,
        deterministic_fallback: bool,
    ) -> tuple[bool, bool, list[str]]:
        """Run 2-pass order-shuffled correctness check when key_fact_recall is ambiguous."""
        tpl = load_prompt("judge_correctness.v1.md")

        def _call(ordered_facts: list[str]) -> bool:
            facts_block = (
                "\n".join(f"- {f}" for f in ordered_facts)
                if ordered_facts
                else "(No key facts provided; check whether answer directly addresses question.)"
            )
            res = self.llm_client.json(
                tpl.format(
                    question=question,
                    key_facts_block=facts_block,
                    answer=answer_text,
                ),
                schema=_CorrectnessVerdict,
                model=self._judge_model(),
                temperature=0.0,
                seed=self.seed,
                role="judge",
                max_repairs=2,
            )
            return bool(res.correct)

        try:
            first = _call(key_facts)
            second_order = list(reversed(key_facts)) if len(key_facts) >= 2 else list(key_facts)
            second = _call(second_order) if len(key_facts) >= 2 else first
        except LLMProviderError:
            raise
        except Exception as exc:
            return (
                deterministic_fallback,
                True,
                [f"LLM correctness check failed ({exc}); deterministic fallback."],
            )

        if first != second:
            return (
                deterministic_fallback,
                True,
                ["Position bias disagreement in LLM correctness; deterministic fallback."],
            )
        return first, False, []

    def evaluate_case(
        self,
        case: EvalCase,
        record: AnswerRecord,
        chunk_map: dict[str, Chunk],
    ) -> CaseVerdict:
        """Score a single AnswerRecord against an EvalCase."""
        out = record.output
        reasons: list[str] = []
        uncertain = False
        pos_disagreement = False

        # 1. Authoritative deterministic refusal check
        refusal_ok, ref_reason = check_refusal(case.expected_behavior, out.refused)
        if not refusal_ok:
            reasons.append(ref_reason)

        # 2. Authoritative deterministic citation check
        citations_valid, cit_errors = check_citation_validity(record, chunk_map)
        if not citations_valid:
            reasons.extend(cit_errors)

        # 3. Deterministic numeric hallucination guard
        retrieved_chunks = [
            chunk_map[rc.chunk_id]
            for rc in record.retrieved_chunks
            if rc.chunk_id in chunk_map
        ]
        retrieved_texts = [c.text for c in retrieved_chunks]
        tool_texts = [
            json.dumps(tc.result, ensure_ascii=False)
            for tc in record.tool_calls
            if tc.ok and tc.result is not None
        ]
        ungrounded_nums = (
            []
            if out.refused
            else find_ungrounded_numbers(
                out.answer, retrieved_texts, tool_texts, case.question
            )
        )
        if ungrounded_nums:
            reasons.append(
                f"Ungrounded numeric value(s) not in context/tools: {ungrounded_nums}"
            )

        # 4. Deterministic canonical key-fact recall
        recall, _, missed_facts = compute_key_fact_recall(case.key_facts, out.answer)
        if case.expected_behavior == "answer" and missed_facts:
            reasons.append(f"Missed key facts: {missed_facts}")

        # 5. Groundedness verdict (deterministic first, LLM claim-level second)
        deterministic_grounded = citations_valid and (len(ungrounded_nums) == 0)
        claim_results: list[ClaimVerification] = []
        if out.refused:
            grounded = citations_valid
        elif not deterministic_grounded:
            grounded = False
        else:
            ev_parts = [
                f"[{c.doc_id} | {c.chunk_id}]\n{c.text}" for c in retrieved_chunks
            ] + [
                f"[tool:{tc.tool_name}]\n{json.dumps(tc.result)}"
                for tc in record.tool_calls
                if tc.ok
            ]
            grounded, claim_results, claim_uncertain, claim_notes = (
                self._judge_claim_groundedness(
                    out.answer,
                    "\n\n".join(ev_parts) or "(no retrieved evidence)",
                    retrieved_texts + tool_texts,
                    deterministic_ok=deterministic_grounded,
                )
            )
            reasons.extend(claim_notes)
            if claim_uncertain:
                uncertain = True
                pos_disagreement = True

        # 6. Correctness verdict (deterministic first, LLM only when ambiguous)
        if case.expected_behavior == "refuse":
            correct = refusal_ok
        elif not refusal_ok:
            correct = False
        elif case.key_facts:
            if recall == 1.0:
                correct = True
            elif recall == 0.0:
                correct = False
            else:
                correct, corr_uncertain, corr_notes = self._judge_ambiguous_correctness(
                    case.question,
                    case.key_facts,
                    out.answer,
                    deterministic_fallback=(recall >= 0.5),
                )
                reasons.extend(corr_notes)
                if corr_uncertain:
                    uncertain = True
                    pos_disagreement = True
        else:
            # Clarify / factless cases
            correct, corr_uncertain, corr_notes = self._judge_ambiguous_correctness(
                case.question,
                [],
                out.answer,
                deterministic_fallback=(refusal_ok and citations_valid),
            )
            reasons.extend(corr_notes)
            if corr_uncertain:
                uncertain = True
                pos_disagreement = True

        passed = refusal_ok and grounded and correct
        if passed and not reasons:
            reasons.append("All deterministic and claim-level checks passed.")

        return CaseVerdict(
            case_id=case.id,
            category=case.category,
            expected_behavior=case.expected_behavior,
            grounded=grounded,
            correct=correct,
            refusal_ok=refusal_ok,
            passed=passed,
            reasons=reasons,
            uncertain=uncertain,
            key_fact_recall=recall,
            citations_valid=citations_valid,
            ungrounded_numbers=ungrounded_nums,
            claim_results=claim_results,
            position_bias_disagreement=pos_disagreement,
            answer_length_chars=len(out.answer),
        )

    def evaluate_suite(
        self,
        definition: AgentDefinition,
        cases: list[EvalCase],
        records: list[AnswerRecord],
        chunk_map: dict[str, Chunk],
        *,
        split: str = "dev",
        out_dir: Path | str | None = None,
    ) -> EvalScorecard:
        """Evaluate a full list of cases/records and compute scorecard metrics."""
        if len(cases) != len(records):
            raise ValueError("cases and records must have identical length.")

        results: list[EvalCaseResult] = []
        by_category: dict[str, list[CaseVerdict]] = {}
        lengths: list[float] = []
        passes: list[float] = []

        for case, record in zip(cases, records, strict=True):
            verdict = self.evaluate_case(case, record, chunk_map)
            results.append(
                EvalCaseResult(
                    case_id=case.id,
                    category=case.category,
                    split=case.split,
                    question=case.question,
                    expected_behavior=case.expected_behavior,
                    answer_record=record,
                    verdict=verdict,
                )
            )
            by_category.setdefault(case.category, []).append(verdict)
            lengths.append(float(verdict.answer_length_chars))
            passes.append(1.0 if verdict.passed else 0.0)

        cat_metrics: list[CategoryMetric] = []
        for cat in sorted(by_category):
            items = by_category[cat]
            n = len(items)
            p = sum(1 for v in items if v.passed)
            lo, hi = wilson_interval(p, n)
            cat_metrics.append(
                CategoryMetric(
                    category=cat,
                    total=n,
                    passed=p,
                    pass_rate=round(p / n, 4) if n else 0.0,
                    wilson_low=lo,
                    wilson_high=hi,
                )
            )

        total = len(results)
        passed_total = sum(1 for r in results if r.verdict.passed)
        overall_rate = round(passed_total / total, 4) if total else 0.0
        o_low, o_high = wilson_interval(passed_total, total)

        judge_m = self._judge_model()
        agent_m = definition.model.name
        same_model = judge_m.strip().lower() == agent_m.strip().lower()
        same_family = model_family(judge_m) == model_family(agent_m)

        suspicion_flag = total > 0 and passed_total == total
        suspicion_note = (
            "100% pass rate detected — evaluation set may be too easy, incomplete, or overfit."
            if suspicion_flag
            else None
        )

        scorecard = EvalScorecard(
            split=split,
            agent_name=definition.meta.name,
            agent_model=agent_m,
            judge_model=judge_m,
            same_model_as_agent=same_model,
            same_family_as_agent=same_family,
            total_cases=total,
            passed_cases=passed_total,
            overall_pass_rate=overall_rate,
            overall_wilson_low=o_low,
            overall_wilson_high=o_high,
            suspicion_flag=suspicion_flag,
            suspicion_note=suspicion_note,
            length_pass_correlation=pearson_correlation(lengths, passes),
            category_metrics=cat_metrics,
            results=results,
        )

        if out_dir is not None:
            scorecard.save(out_dir)

        return scorecard
