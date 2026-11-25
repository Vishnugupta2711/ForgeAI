"""Unit tests for forge.judge (deterministic checks, bias defences, canary selftest)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import (
    AgentOutput,
    AnswerRecord,
    Citation,
    RetrievedChunkRecord,
)
from forge.cli import app
from forge.config import ForgeConfig
from forge.evals import EvalCase, EvalProvenance, EvalSet, EvalSourceRef
from forge.ingest import ingest_folder
from forge.judge import (
    Judge,
    check_citation_validity,
    compute_key_fact_recall,
    find_ungrounded_numbers,
    run_judge_selftest,
    wilson_interval,
)
from forge.llm import FakeLLM, LLMMessage

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"
HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def _judge_fake_responder(chunk_texts: list[str], *, fabricate_quote: bool = False):
    """Callback simulating Judge LLM responses for claim extraction, verification, correctness."""

    def _respond(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Extract all atomic factual claims" in prompt:
            ans_part = prompt.split("Answer:\n", 1)[1].split("\n\nReturn ONLY", 1)[0].strip()
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", ans_part)
                if len(s.strip()) > 6
                and "thank you" not in s.lower()
                and "please consult" not in s.lower()
            ]
            return json.dumps({"claims": sentences or [ans_part]})

        if "Verify whether EACH numbered claim" in prompt:
            claims_section = prompt.split("Claims to verify:\n", 1)[1].split(
                "\n\nEvidence", 1
            )[0]
            evidence_section = prompt.split("Evidence (Retrieved Passages & Tool Outputs):\n", 1)[
                1
            ].split("\n\nReturn ONLY", 1)[0]
            verdicts = []
            for line in claims_section.splitlines():
                m = re.match(r"^(\d+)\.\s*(.+)$", line.strip())
                if not m:
                    continue
                idx = int(m.group(1))
                claim = m.group(2)
                if fabricate_quote:
                    verdicts.append(
                        {
                            "claim_index": idx,
                            "supported": True,
                            "quote": "Completely fabricated quote not in evidence.",
                        }
                    )
                    continue
                # Find verbatim support in chunk_texts / evidence
                matched_quote = ""
                for txt in chunk_texts:
                    for sub_line in txt.splitlines():
                        s_clean = sub_line.strip()
                        if s_clean and (
                            s_clean.lower() in claim.lower()
                            or claim.lower() in s_clean.lower()
                            or any(
                                tok in s_clean.lower()
                                for tok in re.findall(r"\b\d+\b", claim)
                            )
                        ):
                            if "override code" not in claim.lower():
                                matched_quote = s_clean
                                break
                    if matched_quote:
                        break
                if matched_quote and matched_quote in evidence_section:
                    verdicts.append(
                        {"claim_index": idx, "supported": True, "quote": matched_quote}
                    )
                else:
                    verdicts.append({"claim_index": idx, "supported": False, "quote": ""})
            return json.dumps({"verdicts": verdicts})

        if "Determine whether the assistant's answer correctly covers" in prompt:
            return json.dumps({"correct": True, "reason": "Covered key facts."})

        raise ValueError(f"Unexpected prompt in judge test: {prompt[:120]}")

    return _respond


def test_deterministic_checks_canonicalization_and_numeric_guard() -> None:
    recall, matched, missed = compute_key_fact_recall(
        ["Deploy pods at 45 meters", "Budget is USD 1200 on 2026-06-15"],
        "Pods go at 45m and budget is $1,200.00 on June 15, 2026.",
    )
    assert recall == 1.0
    assert len(matched) == 2
    assert missed == []

    # Numeric guard catches hallucinated numbers, allows retrieved & tool numbers
    ungrounded = find_ungrounded_numbers(
        "Depth is 45 meters and backup lasts 999 hours with total 225.",
        retrieved_texts=["Deploy hydrophone pods at 45 meters."],
        tool_outputs=['{"result": 225}'],
        question_text="What depth and total?",
    )
    assert ungrounded == ["999"]

    lo, hi = wilson_interval(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0


def test_citation_validity_requires_actually_retrieved_chunks() -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    chunk_map = {c.chunk_id: c for c in chunks}
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    buoy = next(c for c in chunks if "Depot North" in c.text)

    # Valid citation from retrieved chunk
    ok_rec = AnswerRecord(
        question="Depth?",
        output=AgentOutput(
            answer="45 meters.",
            citations=[
                Citation(
                    doc_id=acoustic.doc_id,
                    chunk_id=acoustic.chunk_id,
                    quote="Deploy hydrophone pods at 45 meters",
                )
            ],
            refused=False,
        ),
        retrieved_chunks=[
            RetrievedChunkRecord(chunk_id=acoustic.chunk_id, doc_id=acoustic.doc_id, score=0.016)
        ],
    )
    valid, errs = check_citation_validity(ok_rec, chunk_map)
    assert valid and not errs

    # Unretrieved chunk quote must be rejected!
    unretrieved_rec = ok_rec.model_copy(
        update={
            "output": AgentOutput(
                answer="Depot North.",
                citations=[
                    Citation(
                        doc_id=buoy.doc_id,
                        chunk_id=buoy.chunk_id,
                        quote="Sensors undergo salt-fog calibration every 90 days at Depot North",
                    )
                ],
                refused=False,
            )
        }
    )
    bad_valid, bad_errs = check_citation_validity(unretrieved_rec, chunk_map)
    assert not bad_valid
    assert bad_errs


def test_judge_rejects_fabricated_quote_and_handles_position_bias_uncertainty() -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    chunk_map = {c.chunk_id: c for c in chunks}
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)

    case = EvalCase(
        id="DEV-001",
        category="single_doc",
        question="What depth are hydrophone pods deployed?",
        expected_behavior="answer",
        expected_sources=[EvalSourceRef(doc_id=acoustic.doc_id, chunk_id=acoustic.chunk_id)],
        key_facts=["Deploy hydrophone pods at 45 meters below mean sea level"],
        provenance=prov,
    )
    record = AnswerRecord(
        question=case.question,
        output=AgentOutput(
            answer="Deploy hydrophone pods at 45 meters below mean sea level.",
            citations=[
                Citation(
                    doc_id=acoustic.doc_id,
                    chunk_id=acoustic.chunk_id,
                    quote="Deploy hydrophone pods at 45 meters below mean sea level",
                )
            ],
            refused=False,
            confidence=0.9,
        ),
        retrieved_chunks=[
            RetrievedChunkRecord(chunk_id=acoustic.chunk_id, doc_id=acoustic.doc_id, score=0.016)
        ],
    )

    # 1. Judge LLM fabricates a supporting quote -> claim rejected -> grounded=False
    fab_llm = FakeLLM(responses=[_judge_fake_responder([acoustic.text], fabricate_quote=True)] * 5)
    verdict_fab = Judge(fab_llm).evaluate_case(case, record, chunk_map)
    assert verdict_fab.grounded is False
    assert verdict_fab.passed is False

    # 2. Position bias disagreement on 2 claims -> uncertain=True and deterministic fallback!
    two_claim_record = record.model_copy(
        update={
            "output": AgentOutput(
                answer=(
                    "Deploy hydrophone pods at 45 meters below mean sea level. "
                    "Avoid anchoring within 200 meters of active lanes."
                ),
                citations=[
                    Citation(
                        doc_id=acoustic.doc_id,
                        chunk_id=acoustic.chunk_id,
                        quote="Deploy hydrophone pods at 45 meters below mean sea level",
                    )
                ],
                refused=False,
                confidence=0.9,
            )
        }
    )
    disagree_llm = FakeLLM(
        responses=[
            json.dumps({"claims": ["Claim A", "Claim B"]}),
            json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_index": 1,
                            "supported": True,
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        },
                        {
                            "claim_index": 2,
                            "supported": True,
                            "quote": "Avoid anchoring within 200 meters",
                        },
                    ]
                }
            ),
            # Second shuffled pass disagrees on one claim!
            json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_index": 1,
                            "supported": True,
                            "quote": "Avoid anchoring within 200 meters",
                        },
                        {"claim_index": 2, "supported": False, "quote": ""},
                    ]
                }
            ),
        ]
    )
    verdict_uncertain = Judge(disagree_llm).evaluate_case(case, two_claim_record, chunk_map)
    assert verdict_uncertain.uncertain is True
    assert verdict_uncertain.position_bias_disagreement is True
    # Deterministic fallback passes because citation is valid and no ungrounded numbers exist
    assert verdict_uncertain.grounded is True
    assert verdict_uncertain.passed is True


def test_judge_selftest_all_eight_canary_types(tmp_path: Path) -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    chunk_map = {c.chunk_id: c for c in chunks}
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    buoy = next(c for c in chunks if "Depot North" in c.text)

    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)
    eval_set = EvalSet(
        agent_name="test-agent",
        eval_writer_model="llama3.1:8b",
        agent_model="qwen2.5:3b-instruct",
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[
            EvalCase(
                id="DEV-001",
                category="single_doc",
                question="At what depth should hydrophone pods be deployed?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=acoustic.doc_id, chunk_id=acoustic.chunk_id)
                ],
                key_facts=["Deploy hydrophone pods at 45 meters below mean sea level"],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-002",
                category="single_doc",
                question="How often are sensors calibrated at Depot North?",
                expected_behavior="answer",
                expected_sources=[EvalSourceRef(doc_id=buoy.doc_id, chunk_id=buoy.chunk_id)],
                key_facts=["Sensors undergo salt-fog calibration every 90 days at Depot North."],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-003",
                category="out_of_scope",
                question="What is the classified HR compensation band?",
                expected_behavior="refuse",
                provenance=prov,
            ),
        ],
    )

    llm = FakeLLM(responses=[_judge_fake_responder([c.text for c in chunks])] * 100)
    out_file = tmp_path / "judge_reliability.json"
    report = run_judge_selftest(llm, chunk_map, eval_set=eval_set, out_path=out_file)

    assert out_file.is_file()
    canary_types = {m.canary_type for m in report.per_canary_type}
    assert canary_types == {
        "gold_answer",
        "numerically_corrupted",
        "hallucinated_claim",
        "wrong_document",
        "padded_verbose",
        "terse_correct",
        "correct_refusal",
        "wrongful_refusal",
    }
    assert report.overall_false_pass_rate == 0.0
    assert report.overall_false_fail_rate == 0.0


def test_cli_evals_run_and_judge_selftest(tmp_path: Path, monkeypatch) -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)

    eval_set = EvalSet(
        agent_name="marine-sensor-assistant",
        eval_writer_model="llama3.1:8b",
        agent_model="qwen2.5:3b-instruct",
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[
            EvalCase(
                id="DEV-001",
                category="single_doc",
                question="What depth should hydrophone pods be deployed?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=acoustic.doc_id, chunk_id=acoustic.chunk_id)
                ],
                key_facts=["Deploy hydrophone pods at 45 meters below mean sea level"],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-002",
                category="out_of_scope",
                question="What is the executive bonus budget?",
                expected_behavior="refuse",
                provenance=prov,
            ),
        ],
    )
    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text(eval_set.model_dump_json(indent=2), encoding="utf-8")

    judge_cb = _judge_fake_responder([c.text for c in chunks])

    def _combined_responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "User question:" in prompt:
            if "executive bonus" in prompt.lower():
                return json.dumps(
                    {
                        "action": "final",
                        "answer": "The documents do not cover executive bonuses.",
                        "citations": [],
                        "refused": True,
                        "refusal_reason": "Out of scope.",
                        "confidence": 0.95,
                    }
                )
            return json.dumps(
                {
                    "action": "final",
                    "answer": "Deploy hydrophone pods at 45 meters below mean sea level.",
                    "citations": [
                        {
                            "doc_id": acoustic.doc_id,
                            "chunk_id": acoustic.chunk_id,
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.95,
                }
            )
        return judge_cb(messages)

    fake_llm = FakeLLM(responses=[_combined_responder] * 100, config=ForgeConfig())
    monkeypatch.setattr(
        forge_cli.LLMClient,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: fake_llm),
    )

    runner = CliRunner()
    eval_out = tmp_path / "eval_run"
    res_run = runner.invoke(
        app,
        [
            "evals",
            "run",
            "--definition",
            str(HAND_AGENT_PATH),
            "--eval-set",
            str(eval_path),
            "--docs",
            str(DOCS_DIR),
            "--out",
            str(eval_out),
            "--bm25-only",
        ],
    )
    assert res_run.exit_code == 0
    assert (eval_out / "results.json").is_file()
    assert (eval_out / "scorecard.md").is_file()
    scorecard_md = (eval_out / "scorecard.md").read_text(encoding="utf-8")
    assert "SUSPICION FLAG" in scorecard_md

    self_out = tmp_path / "selftest_out"
    res_self = runner.invoke(
        app,
        [
            "judge-selftest",
            "--eval-set",
            str(eval_path),
            "--docs",
            str(DOCS_DIR),
            "--out",
            str(self_out),
        ],
    )
    assert res_self.exit_code == 0
    assert (self_out / "judge_reliability.json").is_file()
