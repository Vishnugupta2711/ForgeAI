"""Unit tests for Phase 9d-1: deterministic-first judge, citations, chunker, and report gate."""

from __future__ import annotations

import json
from pathlib import Path

from forge.agent import AgentOutput, AnswerRecord, Citation, RetrievedChunkRecord
from forge.evals import EvalCase, EvalProvenance, EvalSet, EvalSourceRef
from forge.groundability import GroundabilityGuard
from forge.ingest import Chunk, Document, chunk_document
from forge.judge import Judge, JudgeReliabilityReport, check_citation_validity, run_judge_selftest
from forge.llm import FakeLLM, LLMMessage
from forge.report import build_report
from forge.retrieval import HybridIndex

_CSV_ROW_TEXT = (
    "part_no: SP-1026\n"
    "description: Refrigerant R404A Service Cylinder MK-2\n"
    "unit_price_usd: 257.50\n"
    "min_order_qty: 2\n"
    "lead_time_days: 10\n"
    "stock_location: Rotterdam-WH1\n"
    "hazmat_flag: true"
)


def _csv_chunk_map() -> dict[str, Chunk]:
    ch = Chunk(
        chunk_id="spares_catalogue.csv::row-26",
        doc_id="spares_catalogue.csv",
        text=_CSV_ROW_TEXT,
        section_heading="Row 26",
        metadata={"doc_type": "csv", "row_index": 26},
    )
    other = Chunk(
        chunk_id="spares_catalogue.csv::row-16",
        doc_id="spares_catalogue.csv",
        text=(
            "part_no: SP-1016\n"
            "description: Refrigerant R404A Service Cylinder MK-2\n"
            "unit_price_usd: 220.00\n"
            "min_order_qty: 2\n"
            "lead_time_days: 8\n"
            "stock_location: Busan-WH4\n"
            "hazmat_flag: true"
        ),
        section_heading="Row 16",
        metadata={"doc_type": "csv", "row_index": 16},
    )
    return {ch.chunk_id: ch, other.chunk_id: other}


def test_c1_citation_validation_csv_segments_and_rejections() -> None:
    chunk_map = _csv_chunk_map()
    cited = chunk_map["spares_catalogue.csv::row-26"]

    def _record_with_quote(quote: str, *, retrieved_ids: list[str] | None = None) -> AnswerRecord:
        ids = retrieved_ids if retrieved_ids is not None else [cited.chunk_id]
        return AnswerRecord(
            question="What is the unit price of Refrigerant R404A Service Cylinder MK-2?",
            output=AgentOutput(
                answer="The unit price is $257.50.",
                citations=[Citation(doc_id=cited.doc_id, chunk_id=cited.chunk_id, quote=quote)],
                refused=False,
                confidence=0.95,
            ),
            retrieved_chunks=[
                RetrievedChunkRecord(chunk_id=cid, doc_id=chunk_map[cid].doc_id, score=0.016)
                for cid in ids
            ],
        )

    # Real HELD-005 non-contiguous multiline CSV quote must pass
    ok, errs = check_citation_validity(
        _record_with_quote("part_no: SP-1026\nunit_price_usd: 257.50"), chunk_map
    )
    assert ok and not errs

    # Ellipsis & pipe variants must also pass
    ok_ell, _ = check_citation_validity(
        _record_with_quote("part_no: SP-1026 ... unit_price_usd: 257.50"), chunk_map
    )
    assert ok_ell
    ok_pipe, _ = check_citation_validity(
        _record_with_quote("part_no: SP-1026 | unit_price_usd: 257.50"), chunk_map
    )
    assert ok_pipe

    # Altered number in HELD-005 quote must FAIL
    bad_num, bad_num_errs = check_citation_validity(
        _record_with_quote("part_no: SP-1026\nunit_price_usd: 258.50"), chunk_map
    )
    assert not bad_num and bad_num_errs

    # Fabricated segment alongside a real segment must FAIL
    bad_seg, bad_seg_errs = check_citation_validity(
        _record_with_quote("part_no: SP-1026\npriority_tier: 99"), chunk_map
    )
    assert not bad_seg and bad_seg_errs

    # Cited chunk never retrieved must FAIL
    bad_ret, bad_ret_errs = check_citation_validity(
        _record_with_quote(
            "part_no: SP-1026\nunit_price_usd: 257.50",
            retrieved_ids=["spares_catalogue.csv::row-16"],
        ),
        chunk_map,
    )
    assert not bad_ret and bad_ret_errs


def test_k1_chunker_snaps_boundaries_no_midword_cut() -> None:
    text = (
        "3. Weekly Consolidation Cutoffs and Port Exceptions\n"
        "Inter-port sea consolidation shuttles close every Thursday at 16:00 UTC across all hubs. "
        "Transfer orders confirmed after Thursday 16:00 UTC wait for the next weekly cycle. "
        "Local same-port collections dispatch daily and do not wait for the cutoff. "
        "European hub exception: Road shuttle transfers exclusively between Rotterdam-WH1 and "
        "Hamburg-WH3 for non-hazmat orders under 50 kg reduce transfer days down to 1 day. "
        "Seasonal congestion buffer: During peak dry-docking windows (March 1 to April 30 and "
        "September 1 to October 31), standard sea dispatches out of Singapore-WH2 and\n"
        "Busan-WH4 add +2 calendar days to effective_lead_days."
    )
    doc = Document(doc_id="logistics.pdf", path="/tmp/logistics.pdf", type="pdf", text=text)
    chunks = chunk_document(doc, chunk_size=180, overlap=40)
    assert len(chunks) >= 3
    for ch in chunks:
        # Must never start with a chopped word fragment like "ry-docking"
        assert not ch.text.startswith("ry-docking")
        first_token = ch.text.split()[0]
        assert first_token in text.split() or any(
            line.strip().startswith(first_token) for line in text.splitlines()
        )


def test_k2_groundability_verifies_quotes_across_adjacent_chunks() -> None:
    c0 = Chunk(
        chunk_id="logistics.pdf::c1",
        doc_id="logistics.pdf",
        text="1. General Maritime Logistics Overview for Port Deliveries.",
    )
    c1 = Chunk(
        chunk_id="logistics.pdf::c4",
        doc_id="logistics.pdf",
        text=(
            "Seasonal congestion buffer: During peak dry-docking windows "
            "(March 1 to April 30 and September 1 to October 31), "
            "standard sea dispatches out of Singapore-WH2 and"
        ),
    )
    c2 = Chunk(
        chunk_id="logistics.pdf::c5",
        doc_id="logistics.pdf",
        text="Busan-WH4 add +2 calendar days to effective_lead_days.",
    )
    index = HybridIndex.build([c0, c1, c2])
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "status": "SUPPORTED",
                    "quotes": [
                        {
                            "chunk_id": "logistics.pdf::c4",
                            "quote": (
                                "During peak dry-docking windows (March 1 to April 30 and "
                                "September 1 to October 31), standard sea dispatches out of "
                                "Singapore-WH2 and Busan-WH4 add +2 calendar days to "
                                "effective_lead_days."
                            ),
                        }
                    ],
                    "reasoning": "Explicit regional congestion exception.",
                    "gap_note": None,
                }
            )
        ]
    )
    from forge.groundability import AtomicTopic

    guard = GroundabilityGuard(index, llm, top_k=2, alpha=0.0)
    verdict = guard.verify_topic(
        AtomicTopic(
            topic_id="TOP-008",
            title="Regional Congestion Windows",
            description=(
                "Dry-docking windows out of Singapore-WH2 and Busan-WH4 effective_lead_days"
            ),
        )
    )
    assert verdict.status == "SUPPORTED"
    assert verdict.quote_verification_failed is False
    assert any(q.verified_substring for q in verdict.quotes)


def test_j1_empty_quote_and_prompt_leak_fall_back_and_catches_hallucination() -> None:
    chunk_map = _csv_chunk_map()
    row = chunk_map["spares_catalogue.csv::row-26"]
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)
    case = EvalCase(
        id="DEV-001",
        category="single_doc",
        question="What is SP-1026 unit price?",
        expected_behavior="answer",
        expected_sources=[EvalSourceRef(doc_id=row.doc_id, chunk_id=row.chunk_id)],
        key_facts=["unit_price_usd: 257.50"],
        provenance=prov,
    )
    base_record = AnswerRecord(
        question=case.question,
        output=AgentOutput(
            # Middle-band paraphrase of row-26 so it goes to LLM claim verifier
            answer=(
                "The catalogued unit price USD for SP-1026 equals 257.50 "
                "at Rotterdam-WH1 harbor depot today."
            ),
            citations=[
                Citation(
                    doc_id=row.doc_id,
                    chunk_id=row.chunk_id,
                    quote="part_no: SP-1026\nunit_price_usd: 257.50",
                )
            ],
            refused=False,
        ),
        retrieved_chunks=[
            RetrievedChunkRecord(chunk_id=row.chunk_id, doc_id=row.doc_id, score=0.016)
        ],
    )

    # Empty quote / prompt leak on supported=False treats claim as UNCERTAIN
    leak_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_index": 1,
                            "supported": False,
                            "quote": "Return ONLY a JSON object:",
                            "reason": "",
                        }
                    ]
                }
            )
        ]
    )
    v_unc = Judge(leak_llm).evaluate_case(case, base_record, chunk_map)
    assert v_unc.grounded is True
    assert v_unc.passed is True
    assert v_unc.used_fallback is True
    assert v_unc.passed_via_fallback is True
    assert v_unc.uncertain_claims == 1

    # Hallucinated plausible sentence with zero numbers is caught deterministically
    hall_record = base_record.model_copy(
        update={
            "output": AgentOutput(
                answer=(
                    "For SP-1026, the unit price usd is 257.50. "
                    "Emergency maritime clearance protocol requires dual harbor authority sign-off."
                ),
                citations=base_record.output.citations,
                refused=False,
            )
        }
    )
    v_hall = Judge(FakeLLM(responses=[])).evaluate_case(case, hall_record, chunk_map)
    assert v_hall.grounded is False
    assert v_hall.passed is False


def test_j2_csv_canaries_use_natural_sentences_and_clean_pass_reporting(tmp_path: Path) -> None:
    chunk_map = _csv_chunk_map()
    row = chunk_map["spares_catalogue.csv::row-26"]
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)
    eval_set = EvalSet(
        agent_name="test-spares",
        eval_writer_model="llama3.1:8b",
        agent_model="qwen2.5:3b-instruct",
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[
            EvalCase(
                id="DEV-001",
                category="single_doc",
                question="What is the unit price for SP-1026?",
                expected_behavior="answer",
                expected_sources=[EvalSourceRef(doc_id=row.doc_id, chunk_id=row.chunk_id)],
                key_facts=["unit_price_usd: 257.50"],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-002",
                category="out_of_scope",
                question="What is the company cafeteria menu?",
                expected_behavior="refuse",
                provenance=prov,
            ),
        ],
    )

    def _middle_ok(_msgs: list[LLMMessage]) -> str:
        return json.dumps(
            {"verdicts": [{"claim_index": 1, "supported": True, "quote": _CSV_ROW_TEXT}]}
        )

    rel = run_judge_selftest(
        FakeLLM(responses=[_middle_ok] * 20),
        chunk_map,
        eval_set=eval_set,
        out_path=tmp_path / "judge_reliability.json",
    )
    assert rel.overall_false_fail_rate == 0.0
    assert rel.overall_false_pass_rate == 0.0
    assert rel.passed_via_fallback_count == 0
    assert rel.passed_cleanly_count > 0
    assert rel.overall_clean_pass_rate == 1.0


def test_j3_report_unreliable_judge_gate_banner_and_narrative_block(tmp_path: Path) -> None:
    from forge.judge import CaseVerdict, EvalCaseResult, EvalScorecard

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rel = JudgeReliabilityReport(
        judge_model="llama3.1:8b",
        total_canaries=16,
        overall_false_pass_rate=0.0,
        overall_false_fail_rate=0.375,
        per_canary_type=[],
        cases=[],
    )
    rel.save(run_dir / "judge_reliability.json")

    bad_rec = AnswerRecord(
        question="Q?",
        output=AgentOutput(answer="bad", citations=[], refused=False),
        retrieved_chunks=[],
    )
    bad_verdict = CaseVerdict(
        case_id="DEV-001",
        category="single_doc",
        expected_behavior="answer",
        grounded=False,
        correct=False,
        refusal_ok=True,
        passed=False,
        reasons=["Missing citations."],
    )
    scorecard = EvalScorecard(
        split="dev",
        agent_name="test-agent",
        agent_model="qwen2.5:3b-instruct",
        judge_model="llama3.1:8b",
        same_model_as_agent=False,
        same_family_as_agent=False,
        total_cases=1,
        passed_cases=0,
        overall_pass_rate=0.0,
        overall_wilson_low=0.0,
        overall_wilson_high=0.7935,
        results=[
            EvalCaseResult(
                case_id="DEV-001",
                category="single_doc",
                split="dev",
                question="Q?",
                expected_behavior="answer",
                answer_record=bad_rec,
                verdict=bad_verdict,
            )
        ],
    )
    scorecard.save(run_dir / "final_dev_eval")

    bad_score_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "diagnosis_items": [
                        {"case_ids": ["DEV-001"], "diagnosis": "Scored 42.0% overall."}
                    ],
                    "overall_summary": "The agent scored 42.0%.",
                }
            )
        ]
    )
    report_path, _ = build_report(run_dir, llm_client=bad_score_llm)
    text = report_path.read_text(encoding="utf-8")
    assert text.splitlines()[2].startswith("## ⚠️ Judge Reliability Warning")
    assert "unreliable lower bounds" in text.lower()
    assert "42.0%" not in text
    assert "narrative score claims blocked" in text.lower()
