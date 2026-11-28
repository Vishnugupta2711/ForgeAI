"""Two-layer ForgeAI Judge with deterministic checks, claim verification, and bias defences."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import BaseModel, Field

from forge.agent.schema import AgentDefinition, AnswerRecord
from forge.config import ForgeConfig, load_prompt
from forge.evals.generator import model_family
from forge.evals.models import EvalCase
from forge.ingest.models import Chunk
from forge.judge.deterministic import (
    _NUMBER_RE,
    _canon_number,
    canonicalize_text,
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

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
        "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "per",
        "should", "that", "the", "their", "this", "to", "was", "were", "when",
        "where", "which", "who", "will", "with",
    }
)

_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:thank you|thanks)(?:\s+for\s+(?:asking|your question))?[.!]*$", re.I),
    re.compile(r"^(?:i\s+)?hope\s+this\s+helps[.!]*$", re.I),
    re.compile(
        r"^please\s+(?:consult|refer\s+to)\s+the\s+cited\s+passage(?:\s+for\s+reference)?[.!]*$",
        re.I,
    ),
    re.compile(r"^please\s+let\s+me\s+know\s+if\s+you\s+(?:have|need)\s+.*$", re.I),
)

_FRAMING_PREFIX_RE = re.compile(
    r"^(?:after\s+carefully\s+reviewing[^:]*,\s*here\s+is\s+the\s+(?:verified\s+)?summary\s*:\s*"
    r"|here\s+is\s+the\s+(?:verified\s+)?summary\s*:\s*"
    r"|based\s+on\s+(?:the\s+)?(?:documentation|provided\s+documents)\s*:\s*)",
    re.I,
)

_PROMPT_LEAK_MARKERS = (
    "return only a json object",
    "claims to verify",
    "critical rules",
    "claim_index",
    "exact verbatim substring from evidence",
)


def _split_claims_deterministic(answer_text: str) -> list[str]:
    """Split answer into factual sentences deterministically, dropping fixed boilerplate."""
    text = answer_text.strip()
    if not text:
        return []
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    raw_sentences: list[str] = []
    for ln in raw_lines:
        # Strip leading framing wrapper while preserving actual factual sentence
        cleaned_ln = _FRAMING_PREFIX_RE.sub("", ln).strip()
        if not cleaned_ln:
            continue
        parts = re.split(r"(?<=[.!?])\s+", cleaned_ln)
        for part in parts:
            s = _FRAMING_PREFIX_RE.sub("", part.strip()).strip()
            if s:
                raw_sentences.append(s)

    claims: list[str] = []
    for sent in raw_sentences:
        if any(pat.match(sent) for pat in _BOILERPLATE_PATTERNS):
            continue
        claims.append(sent)
    return claims or [text]


def _claim_tokens(text: str) -> list[str]:
    canon = canonicalize_text(text)
    return [t for t in re.findall(r"[a-z0-9]+", canon) if len(t) > 1 and t not in _STOPWORDS]


def _is_prompt_leak(text: str) -> bool:
    low = text.strip().lower()
    return any(marker in low for marker in _PROMPT_LEAK_MARKERS)


def _reason_checks_out(reason: str, claim: str, evidence_texts: list[str]) -> bool:
    cleaned = reason.strip()
    if len(cleaned) < 12 or _is_prompt_leak(cleaned):
        return False
    r_tokens = set(_claim_tokens(cleaned))
    if not r_tokens:
        return False
    c_tokens = set(_claim_tokens(claim))
    if r_tokens & c_tokens:
        return True
    ev_tokens: set[str] = set()
    for ev in evidence_texts:
        ev_tokens.update(_claim_tokens(ev))
    return bool(r_tokens & ev_tokens)


class _ClaimVerdictItem(BaseModel):
    claim_index: int
    supported: bool
    quote: str = ""
    reason: str = ""


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
    ) -> dict[str, tuple[ClaimVerification, _ClaimVerdictItem | None]]:
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
        result: dict[str, tuple[ClaimVerification, _ClaimVerdictItem | None]] = {}
        for idx, claim_text in enumerate(ordered_claims, start=1):
            item = by_index.get(idx)
            if item is None:
                result[claim_text] = (
                    ClaimVerification(
                        claim=claim_text,
                        supported=False,
                        quote="",
                        verified_substring=False,
                        status="uncertain",
                        method="fallback",
                        reason="Missing verifier item.",
                    ),
                    None,
                )
                continue
            leaked = _is_prompt_leak(item.quote) or _is_prompt_leak(item.reason)
            quote_ok = (
                (not leaked) and verify_quote_in_evidence(item.quote, evidence_texts)
                if item.supported
                else False
            )
            result[claim_text] = (
                ClaimVerification(
                    claim=claim_text,
                    supported=bool(item.supported and quote_ok),
                    quote=item.quote,
                    verified_substring=quote_ok,
                    status="supported" if (item.supported and quote_ok) else "unsupported",
                    method="llm",
                    reason=item.reason,
                ),
                item,
            )
        return result

    def _classify_claim_deterministic(
        self,
        claim: str,
        evidence_items: list[str],
    ) -> tuple[str, float, str]:
        """Classify claim into SUPPORTED, UNSUPPORTED, or MIDDLE using single-item overlap."""
        canon_claim = canonicalize_text(claim)
        c_tokens = _claim_tokens(claim)
        c_nums = {_canon_number(n) for n in _NUMBER_RE.findall(canon_claim)}

        best_overlap_any = 0.0
        best_overlap_with_nums = 0.0
        any_ev_has_nums = not c_nums
        best_quote = ""

        for ev in evidence_items:
            canon_ev = canonicalize_text(ev)
            ev_tokens = set(_claim_tokens(ev))
            ev_nums = {_canon_number(n) for n in _NUMBER_RE.findall(canon_ev)}
            nums_ok = c_nums.issubset(ev_nums)
            if nums_ok:
                any_ev_has_nums = True

            if canon_claim and canon_claim in canon_ev:
                overlap = 1.0
            elif c_tokens:
                overlap = sum(1 for t in c_tokens if t in ev_tokens) / len(c_tokens)
            else:
                overlap = 1.0 if (canon_claim and canon_claim in canon_ev) else 0.0

            if overlap > best_overlap_any:
                best_overlap_any = overlap
            if nums_ok and overlap >= best_overlap_with_nums:
                best_overlap_with_nums = overlap
                best_quote = ev.splitlines()[0][:120] if ev.strip() else ""

        support_thresh = self.config.judge.claim_support_threshold
        unsupport_thresh = self.config.judge.claim_unsupport_threshold

        if best_overlap_with_nums >= support_thresh and any_ev_has_nums:
            return "SUPPORTED", best_overlap_with_nums, best_quote
        if best_overlap_any < unsupport_thresh or not any_ev_has_nums:
            return "UNSUPPORTED", best_overlap_any, ""
        return "MIDDLE", best_overlap_with_nums, best_quote

    def _judge_claim_groundedness(
        self,
        answer_text: str,
        evidence_block: str,
        evidence_texts: list[str],
        *,
        deterministic_ok: bool,
        citation_quotes: list[str] | None = None,
    ) -> tuple[bool, list[ClaimVerification], bool, int, bool, list[str]]:
        """Verify claims deterministically first; send only middle band to 2-pass LLM verifier."""
        notes: list[str] = []
        claims = _split_claims_deterministic(answer_text)
        if not claims:
            return deterministic_ok, [], False, 0, False, notes

        det_pool = list(evidence_texts) + [q for q in (citation_quotes or []) if q.strip()]
        middle_claims: list[str] = []
        claim_states: dict[str, ClaimVerification] = {}

        for claim in claims:
            band, overlap, matched_snip = self._classify_claim_deterministic(claim, det_pool)
            if band == "SUPPORTED":
                claim_states[claim] = ClaimVerification(
                    claim=claim,
                    supported=True,
                    quote=matched_snip,
                    verified_substring=bool(matched_snip),
                    status="supported",
                    method="deterministic",
                    reason=f"Deterministic support (overlap={overlap:.2f}).",
                )
            elif band == "UNSUPPORTED":
                claim_states[claim] = ClaimVerification(
                    claim=claim,
                    supported=False,
                    quote="",
                    verified_substring=False,
                    status="unsupported",
                    method="deterministic",
                    reason=f"Deterministic ungrounded claim (overlap={overlap:.2f}).",
                )
            else:
                middle_claims.append(claim)

        uncertain_claims = 0
        used_fallback = False
        pos_disagreement = False

        if middle_claims:
            try:
                pass1 = self._verify_claims_pass(middle_claims, evidence_block, evidence_texts)
            except LLMProviderError:
                raise
            except Exception as exc:
                notes.append(f"Claim verification pass 1 failed ({exc}); deterministic fallback.")
                for mc in middle_claims:
                    uncertain_claims += 1
                    used_fallback = True
                    claim_states[mc] = ClaimVerification(
                        claim=mc,
                        supported=deterministic_ok,
                        quote="",
                        verified_substring=False,
                        status="uncertain",
                        method="fallback",
                        reason=str(exc),
                    )
                pass1 = {}

            pass2: dict[str, tuple[ClaimVerification, _ClaimVerdictItem | None]] = {}
            if pass1 and len(middle_claims) >= 2:
                shuffled = list(reversed(middle_claims))
                if len(middle_claims) > 2:
                    rng = random.Random(self.seed)
                    rng.shuffle(shuffled)
                    if shuffled == middle_claims:
                        shuffled = list(reversed(middle_claims))
                try:
                    pass2 = self._verify_claims_pass(shuffled, evidence_block, evidence_texts)
                except LLMProviderError:
                    raise
                except Exception as exc:
                    notes.append(
                        f"Claim verification pass 2 failed ({exc}); deterministic fallback."
                    )
                    used_fallback = True
                    pass2 = {}

            for mc in middle_claims:
                if mc in claim_states:
                    continue
                v1, item1 = pass1[mc]
                v2_pair = pass2.get(mc)
                if v2_pair is not None and v1.supported != v2_pair[0].supported:
                    pos_disagreement = True
                    uncertain_claims += 1
                    used_fallback = True
                    claim_states[mc] = ClaimVerification(
                        claim=mc,
                        supported=deterministic_ok,
                        quote=v1.quote,
                        verified_substring=v1.verified_substring,
                        status="uncertain",
                        method="fallback",
                        reason="Position bias disagreement between forward and shuffled passes.",
                    )
                    continue

                if item1 is not None and item1.supported:
                    # LLM supported=True requires valid non-leaked verbatim quote
                    if v1.verified_substring:
                        claim_states[mc] = v1
                    else:
                        claim_states[mc] = ClaimVerification(
                            claim=mc,
                            supported=False,
                            quote=item1.quote,
                            verified_substring=False,
                            status="unsupported",
                            method="llm",
                            reason="LLM returned fabricated or unverified supporting quote.",
                        )
                        notes.append("Rejected unsupported or fabricated-quote claim(s).")
                    continue

                # LLM supported=False (or missing item): count only if quote/reason checks out
                q_str = item1.quote.strip() if item1 else ""
                r_str = item1.reason.strip() if item1 else ""
                leaked = (item1 is not None) and (
                    _is_prompt_leak(q_str) or _is_prompt_leak(r_str)
                )
                quote_checks_out = bool(q_str) and (not leaked) and verify_quote_in_evidence(
                    q_str, evidence_texts
                )
                reason_checks_out = (not leaked) and _reason_checks_out(r_str, mc, evidence_texts)

                if quote_checks_out or reason_checks_out:
                    claim_states[mc] = ClaimVerification(
                        claim=mc,
                        supported=False,
                        quote=q_str,
                        verified_substring=quote_checks_out,
                        status="unsupported",
                        method="llm",
                        reason=r_str or "Verifier returned checked-out counter-evidence.",
                    )
                    notes.append("Rejected unsupported claim with checked-out verifier reason.")
                else:
                    uncertain_claims += 1
                    used_fallback = True
                    claim_states[mc] = ClaimVerification(
                        claim=mc,
                        supported=deterministic_ok,
                        quote="" if leaked else q_str,
                        verified_substring=False,
                        status="uncertain",
                        method="fallback",
                        reason=(
                            "LLM unsupported lacked verified quote/reason or leaked prompt; "
                            "deterministic result stands."
                        ),
                    )

        verdicts = [claim_states[c] for c in claims]
        if pos_disagreement:
            notes.append(
                "Position bias disagreement between forward and shuffled claim verification; "
                "falling back to deterministic groundedness."
            )
        elif uncertain_claims > 0:
            notes.append(
                f"{uncertain_claims} middle-band claim(s) uncertain; deterministic stood."
            )

        det_unsupported = [
            v for v in verdicts if v.status == "unsupported" and v.method == "deterministic"
        ]
        if det_unsupported:
            notes.append(
                f"Deterministic guard rejected {len(det_unsupported)} ungrounded claim(s)."
            )

        any_unsupported = any(v.status == "unsupported" for v in verdicts)
        grounded = False if any_unsupported else deterministic_ok
        return grounded, verdicts, pos_disagreement, uncertain_claims, used_fallback, notes

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

        # 5. Groundedness verdict (deterministic first, LLM middle-band claim-level second)
        deterministic_grounded = citations_valid and (len(ungrounded_nums) == 0)
        claim_results: list[ClaimVerification] = []
        uncertain_claims = 0
        used_fallback = False
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
            (
                grounded,
                claim_results,
                claim_pos_bias,
                uncertain_claims,
                used_fallback,
                claim_notes,
            ) = self._judge_claim_groundedness(
                out.answer,
                "\n\n".join(ev_parts) or "(no retrieved evidence)",
                retrieved_texts + tool_texts,
                deterministic_ok=deterministic_grounded,
                citation_quotes=[c.quote for c in out.citations],
            )
            reasons.extend(claim_notes)
            if claim_pos_bias:
                uncertain = True
                pos_disagreement = True
            elif uncertain_claims > 0:
                uncertain = True

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
        passed_via_fallback = passed and (used_fallback or pos_disagreement or uncertain)
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
            uncertain_claims=uncertain_claims,
            used_fallback=used_fallback,
            passed_via_fallback=passed_via_fallback,
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
