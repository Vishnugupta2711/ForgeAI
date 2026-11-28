"""Tests for Phase 9d-1c: verified correctness gate and deterministic CSV claim matching."""

from __future__ import annotations

import json

from forge.agent import AgentOutput, AnswerRecord, Citation, RetrievedChunkRecord
from forge.config import ForgeConfig
from forge.evals import EvalCase, EvalProvenance, EvalSet, EvalSourceRef
from forge.ingest import Chunk
from forge.judge import Judge
from forge.judge.selftest import build_hard_canary_suite
from forge.llm import FakeLLM

_SHIP_ROW_2 = (
    "part_no: SP-1002\n"
    "description: Bilge Oily-Water Sensor Probe MK-3\n"
    "unit_price_usd: 150.00\n"
    "min_order_qty: 2\n"
    "lead_time_days: 7\n"
    "supplier: Generic Marine Supply\n"
    "stock_location: Rotterdam-WH1\n"
    "hazmat_flag: false"
)

_LEAD_TIME_CHUNK = (
    "Orders placed after 14:00 local warehouse time count the following "
    "business day as Day 0 for lead-time tracking."
)


def _ship_chunk_map() -> dict[str, Chunk]:
    c_pdf = Chunk(
        chunk_id="logistics_and_lead_times.pdf::c2",
        doc_id="logistics_and_lead_times.pdf",
        text=_LEAD_TIME_CHUNK,
    )
    c_csv = Chunk(
        chunk_id="spares_catalogue.csv::row-2",
        doc_id="spares_catalogue.csv",
        text=_SHIP_ROW_2,
        metadata={"doc_type": "csv", "row_index": 2},
    )
    return {c_pdf.chunk_id: c_pdf, c_csv.chunk_id: c_csv}


def test_correctness_gate_blocks_bare_true_and_requires_verified_quote_numbers() -> None:
    chunk_map = _ship_chunk_map()
    c_pdf = chunk_map["logistics_and_lead_times.pdf::c2"]
    c_csv = chunk_map["spares_catalogue.csv::row-2"]
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)

    two_fact_case = EvalCase(
        id="DEV-001",
        category="multi_doc",
        question="What is the 14:00 cutoff rule and lead_time_days for SP-1002?",
        expected_behavior="answer",
        expected_sources=[
            EvalSourceRef(doc_id=c_pdf.doc_id, chunk_id=c_pdf.chunk_id),
            EvalSourceRef(doc_id=c_csv.doc_id, chunk_id=c_csv.chunk_id),
        ],
        key_facts=[
            (
                "Orders placed after 14:00 local warehouse time count the following "
                "business day as Day 0 for lead-time tracking."
            ),
            "lead_time_days: 7",
        ],
        provenance=prov,
    )

    # 1. Partial answer (missing lead_time_days: 7) MUST FAIL even if LLM returns bare correct=True
    eval_set = EvalSet(
        agent_name="ship-test",
        eval_writer_model="llama3.1:8b",
        agent_model="qwen2.5:3b-instruct",
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[two_fact_case],
    )
    hard_specs = build_hard_canary_suite(eval_set, chunk_map, max_answer_bases=1)
    partial_spec = next(s for s in hard_specs if s.canary_type == "partial_correct_two_facts")
    bare_true_llm = FakeLLM(
        responses=[json.dumps({"correct": True, "reason": "Mostly answered."})] * 6
    )
    v_partial = Judge(bare_true_llm).evaluate_case(
        partial_spec.eval_case, partial_spec.answer_record, chunk_map
    )
    assert v_partial.key_fact_recall == 0.5
    assert v_partial.correct is False
    assert v_partial.passed is False

    # 2. Surface paraphrase of both facts (missed deterministic token bar) passes ONLY
    # when LLM supplies verified answer substring quote with exact matching numbers.
    paraphrase_rec = AnswerRecord(
        question=two_fact_case.question,
        output=AgentOutput(
            answer=(
                "Orders placed after 14:00 local warehouse time count the following "
                "business day as Day 0 for lead-time tracking. "
                "Dispatch schedules from Rotterdam-WH1 finish inside seven days."
            ),
            citations=[
                Citation(doc_id=c_pdf.doc_id, chunk_id=c_pdf.chunk_id, quote=_LEAD_TIME_CHUNK),
                Citation(doc_id=c_csv.doc_id, chunk_id=c_csv.chunk_id, quote="lead_time_days: 7"),
            ],
            refused=False,
        ),
        retrieved_chunks=[
            RetrievedChunkRecord(chunk_id=c_pdf.chunk_id, doc_id=c_pdf.doc_id, score=0.016),
            RetrievedChunkRecord(chunk_id=c_csv.chunk_id, doc_id=c_csv.doc_id, score=0.015),
        ],
    )
    # Wrong quote numbers ("seven" without numeric 7 -> mismatch) must FAIL
    bad_quote_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "correct": True,
                    "missing_fact_quotes": [
                        {
                            "fact": "lead_time_days: 7",
                            "answer_quote": (
                                "Dispatch schedules from Rotterdam-WH1 "
                                "finish inside seven days."
                            ),
                        }
                    ],
                }
            ),
        ]
        * 4
    )
    # Also test numeric surface match with missing_fact_quotes when fact 2 uses paraphrase tokens
    para_two_case = two_fact_case.model_copy(
        update={
            "key_facts": [
                (
                    "Orders placed after 14:00 local warehouse time count the following "
                    "business day as Day 0 for lead-time tracking."
                ),
                "min_order_qty: 2",
            ]
        }
    )
    good_ans_rec = paraphrase_rec.model_copy(
        update={
            "output": paraphrase_rec.output.model_copy(
                update={
                    "answer": (
                        "Orders placed after 14:00 local warehouse time count the following "
                        "business day as Day 0 for lead-time tracking. "
                        "Shipments for SP-1002 need 2 units at Rotterdam-WH1."
                    ),
                    "citations": [
                        Citation(
                            doc_id=c_pdf.doc_id,
                            chunk_id=c_pdf.chunk_id,
                            quote=_LEAD_TIME_CHUNK,
                        ),
                        Citation(
                            doc_id=c_csv.doc_id,
                            chunk_id=c_csv.chunk_id,
                            quote="part_no: SP-1002\nmin_order_qty: 2",
                        ),
                    ],
                }
            )
        }
    )
    # FakeLLM handles middle-band claim 2 + 2 shuffled correctness passes with verified quote
    good_quote_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "verdicts": [
                        {"claim_index": 1, "supported": True, "quote": "min_order_qty: 2"}
                    ]
                }
            ),
            json.dumps(
                {
                    "correct": True,
                    "missing_fact_quotes": [
                        {
                            "fact": "min_order_qty: 2",
                            "answer_quote": "Shipments for SP-1002 need 2 units at Rotterdam-WH1.",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "correct": True,
                    "missing_fact_quotes": [
                        {
                            "fact": "min_order_qty: 2",
                            "answer_quote": "Shipments for SP-1002 need 2 units at Rotterdam-WH1.",
                        }
                    ],
                }
            ),
        ]
    )
    v_ok = Judge(good_quote_llm).evaluate_case(para_two_case, good_ans_rec, chunk_map)
    assert v_ok.key_fact_recall == 0.5
    assert v_ok.correct is True
    assert v_ok.passed is True

    # Ensure unverified number in missing_fact_quotes still fails
    v_bad = Judge(bad_quote_llm).evaluate_case(two_fact_case, paraphrase_rec, chunk_map)
    assert v_bad.correct is False
    assert v_bad.passed is False


def test_tabular_csv_claim_path_handles_hard_para_and_rejects_wrong_number() -> None:
    cfg = ForgeConfig()
    assert cfg.judge.claim_support_threshold == 0.72
    assert cfg.judge.claim_unsupport_threshold == 0.35

    chunk_map = _ship_chunk_map()
    c_csv = chunk_map["spares_catalogue.csv::row-2"]
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)
    case = EvalCase(
        id="DEV-001",
        category="single_doc",
        question="What is the base lead time for SP-1002?",
        expected_behavior="answer",
        expected_sources=[EvalSourceRef(doc_id=c_csv.doc_id, chunk_id=c_csv.chunk_id)],
        key_facts=["lead_time_days: 7"],
        provenance=prov,
    )

    def _rec(ans: str) -> AnswerRecord:
        return AnswerRecord(
            question=case.question,
            output=AgentOutput(
                answer=ans,
                citations=[
                    Citation(
                        doc_id=c_csv.doc_id,
                        chunk_id=c_csv.chunk_id,
                        quote="part_no: SP-1002\nlead_time_days: 7",
                    )
                ],
                refused=False,
            ),
            retrieved_chunks=[
                RetrievedChunkRecord(chunk_id=c_csv.chunk_id, doc_id=c_csv.doc_id, score=0.016)
            ],
        )

    # Real HARD-PARA-1 claim must pass deterministically via CSV field matching (0 LLM calls!)
    v_ok = Judge(FakeLLM(responses=[]), config=cfg).evaluate_case(
        case, _rec("In addition, base lead time days: 7."), chunk_map
    )
    assert v_ok.grounded is True
    assert v_ok.correct is True
    assert v_ok.passed is True
    assert v_ok.claim_results[0].method == "deterministic_csv"

    # Same claim with wrong number (8) or another field's number (2 from min_order_qty) must FAIL
    v_wrong_8 = Judge(FakeLLM(responses=[]), config=cfg).evaluate_case(
        case, _rec("In addition, base lead time days: 8."), chunk_map
    )
    assert v_wrong_8.grounded is False
    assert v_wrong_8.passed is False

    v_wrong_2 = Judge(FakeLLM(responses=[]), config=cfg).evaluate_case(
        case, _rec("In addition, base lead time days: 2."), chunk_map
    )
    assert v_wrong_2.grounded is False
    assert v_wrong_2.passed is False
