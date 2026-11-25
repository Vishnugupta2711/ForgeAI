"""Unit tests for forge.optimizer (classifier, diff guard, leak guard, rollback, CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import (
    AgentDefinition,
    AgentOutput,
    AnswerRecord,
    RetrievedChunkRecord,
)
from forge.cli import app
from forge.config import ForgeConfig
from forge.evals import EvalCase, EvalProvenance, EvalSet, EvalSourceRef
from forge.ingest import ingest_folder
from forge.judge import CaseVerdict, EvalCaseResult
from forge.llm import FakeLLM, LLMMessage
from forge.optimizer import (
    DiffGuardViolation,
    OptimizerLoop,
    PromptPatch,
    RetrievalPatch,
    apply_patch,
    check_patch_for_leak,
    classify_failure,
    verify_diff_guard,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"
HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def _prov() -> EvalProvenance:
    return EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)


def test_classifier_all_four_categories_and_tiebreaker() -> None:
    prov = _prov()

    # 1. wrong_refusal (missed_refusal)
    oos_case = EvalCase(
        id="DEV-001",
        category="out_of_scope",
        question="What is the CEO salary?",
        expected_behavior="refuse",
        provenance=prov,
    )
    missed_ref_res = EvalCaseResult(
        case_id="DEV-001",
        category="out_of_scope",
        split="dev",
        question=oos_case.question,
        expected_behavior="refuse",
        answer_record=AnswerRecord(
            question=oos_case.question,
            output=AgentOutput(answer="Salary is $100k.", refused=False),
        ),
        verdict=CaseVerdict(
            case_id="DEV-001",
            category="out_of_scope",
            expected_behavior="refuse",
            grounded=True,
            correct=False,
            refusal_ok=False,
            passed=False,
            reasons=["Expected refusal"],
        ),
    )
    f1 = classify_failure(oos_case, missed_ref_res)
    assert f1.failure_category == "wrong_refusal"
    assert f1.subtype == "missed_refusal"

    # 2. retrieval_miss
    ans_case = EvalCase(
        id="DEV-002",
        category="single_doc",
        question="What depth?",
        expected_behavior="answer",
        expected_sources=[
            EvalSourceRef(doc_id="acoustic_guide.md", chunk_id="acoustic_guide.md::c2")
        ],
        key_facts=["45 meters"],
        provenance=prov,
    )
    ret_miss_res = EvalCaseResult(
        case_id="DEV-002",
        category="single_doc",
        split="dev",
        question=ans_case.question,
        expected_behavior="answer",
        answer_record=AnswerRecord(
            question=ans_case.question,
            output=AgentOutput(answer="Wrong chunk.", refused=False),
            retrieved_chunks=[
                RetrievedChunkRecord(
                    chunk_id="field_notes.txt::c1", doc_id="field_notes.txt", score=0.01
                )
            ],
        ),
        verdict=CaseVerdict(
            case_id="DEV-002",
            category="single_doc",
            expected_behavior="answer",
            grounded=False,
            correct=False,
            refusal_ok=True,
            passed=False,
            reasons=["Missed key facts"],
        ),
    )
    f2 = classify_failure(ans_case, ret_miss_res)
    assert f2.failure_category == "retrieval_miss"

    # 3. hallucination (gold chunk retrieved, but ungrounded number/citation)
    hall_res = EvalCaseResult(
        case_id="DEV-002",
        category="single_doc",
        split="dev",
        question=ans_case.question,
        expected_behavior="answer",
        answer_record=AnswerRecord(
            question=ans_case.question,
            output=AgentOutput(answer="Deploy at 45 meters or 999 meters.", refused=False),
            retrieved_chunks=[
                RetrievedChunkRecord(
                    chunk_id="acoustic_guide.md::c2", doc_id="acoustic_guide.md", score=0.016
                )
            ],
        ),
        verdict=CaseVerdict(
            case_id="DEV-002",
            category="single_doc",
            expected_behavior="answer",
            grounded=False,
            correct=True,
            refusal_ok=True,
            passed=False,
            ungrounded_numbers=["999"],
            reasons=["Ungrounded numeric value 999"],
        ),
    )
    f3 = classify_failure(ans_case, hall_res)
    assert f3.failure_category == "hallucination"

    # 4. prompt_defect (gold chunk retrieved, grounded=True, but missed key facts)
    pd_res = EvalCaseResult(
        case_id="DEV-002",
        category="single_doc",
        split="dev",
        question=ans_case.question,
        expected_behavior="answer",
        answer_record=AnswerRecord(
            question=ans_case.question,
            output=AgentOutput(answer="Pods go underwater.", refused=False),
            retrieved_chunks=[
                RetrievedChunkRecord(
                    chunk_id="acoustic_guide.md::c2", doc_id="acoustic_guide.md", score=0.016
                )
            ],
        ),
        verdict=CaseVerdict(
            case_id="DEV-002",
            category="single_doc",
            expected_behavior="answer",
            grounded=True,
            correct=False,
            refusal_ok=True,
            passed=False,
            key_fact_recall=0.0,
            reasons=["Missed key facts"],
        ),
    )
    f4 = classify_failure(ans_case, pd_res)
    assert f4.failure_category == "prompt_defect"


def test_diff_guard_enforces_byte_identical_non_targets() -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    patch = PromptPatch(
        section="grounding_rules",
        op="add_rule",
        text="Always cite exact chunk text.",
        addressed_case_ids=["DEV-001"],
        rationale="Improve citation compliance",
    )
    patched = apply_patch(defn, patch)
    verify_diff_guard(defn, patched, [patch])

    # Mutate an untargeted section -> DiffGuardViolation
    tampered = patched.model_copy(deep=True)
    tampered.system_prompt.role = "Tampered role!"
    with pytest.raises(DiffGuardViolation):
        verify_diff_guard(defn, tampered, [patch])

    # Mutate untargeted retrieval field -> DiffGuardViolation
    ret_patch = RetrievalPatch(target_fields={"top_k": 6}, addressed_case_ids=["DEV-001"])
    ret_patched = apply_patch(defn, ret_patch)
    verify_diff_guard(defn, ret_patched, [ret_patch])
    ret_tampered = ret_patched.model_copy(deep=True)
    ret_tampered.retrieval.hybrid_alpha = 0.9
    with pytest.raises(DiffGuardViolation):
        verify_diff_guard(defn, ret_tampered, [ret_patch])


def test_leak_guard_blocks_eval_overlap_and_allows_general_rules() -> None:
    prov = _prov()
    cases = [
        EvalCase(
            id="DEV-001",
            category="single_doc",
            question="At what depth should hydrophone pods be deployed?",
            expected_behavior="answer",
            key_facts=["Deploy hydrophone pods at 45 meters below mean sea level"],
            provenance=prov,
        )
    ]

    leaky = PromptPatch(
        section="grounding_rules",
        op="add_rule",
        text=(
            "When asked at what depth should hydrophone pods be deployed "
            "answer 45 meters below mean sea level."
        ),
        addressed_case_ids=["DEV-001"],
    )
    ok_leaky, reason = check_patch_for_leak(leaky, cases)
    assert ok_leaky is False
    assert reason is not None

    clean = PromptPatch(
        section="grounding_rules",
        op="add_rule",
        text="State exact measurements and units directly from retrieved excerpts.",
        addressed_case_ids=["DEV-001"],
    )
    ok_clean, _ = check_patch_for_leak(clean, cases)
    assert ok_clean is True


def test_optimizer_rollback_on_regression_3_iter_cap_and_early_stop(tmp_path: Path) -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.retrieval.top_k = 1
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    buoy = next(c for c in chunks if "Depot North" in c.text)
    prov = _prov()

    eval_set = EvalSet(
        agent_name=defn.meta.name,
        eval_writer_model="llama3.1:8b",
        agent_model=defn.model.name,
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[
            EvalCase(
                id="DEV-001",
                category="single_doc",
                split="dev",
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
                category="single_doc",
                split="dev",
                question="How often are sensors calibrated at Depot North?",
                expected_behavior="answer",
                expected_sources=[EvalSourceRef(doc_id=buoy.doc_id, chunk_id=buoy.chunk_id)],
                key_facts=["Sensors undergo salt-fog calibration every 90 days at Depot North."],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-003",
                category="out_of_scope",
                split="dev",
                question="What is the executive salary budget?",
                expected_behavior="refuse",
                provenance=prov,
            ),
        ],
        held_out_cases=[
            EvalCase(
                id="HELD-001",
                category="out_of_scope",
                split="held_out",
                question="What is the secret bonus plan?",
                expected_behavior="refuse",
                provenance=prov,
            )
        ],
    )

    state = {"top_k_seen": 1, "extra_rules_len": 0}

    def _sim_responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Propose at most 2 targeted patches" in prompt:
            # Iteration 1 proposes a RetrievalPatch that regresses; Iter 2 proposes PromptPatch!
            if "retrieval:top_k" not in prompt:
                return json.dumps(
                    {
                        "patches": [
                            {
                                "kind": "retrieval",
                                "target_fields": {"top_k": 2},
                                "addressed_case_ids": ["DEV-002"],
                                "rationale": "Test patch that causes regression in sim.",
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "patches": [
                        {
                            "kind": "prompt",
                            "section": "extra_rules",
                            "op": "add_rule",
                            "text": "Quote exact numeric values from retrieved excerpts.",
                            "addressed_case_ids": ["DEV-002"],
                            "rationale": "Fixes missed numeric calibration details.",
                        }
                    ]
                }
            )

        if "User question:" in prompt:
            # Inspect current system prompt to simulate behavior changes cleanly!
            sys_prompt = messages[0].content
            has_new_rule = "Quote exact numeric values from retrieved excerpts." in sys_prompt
            # Count retrieved chunks in context block
            ctx_chunks = len(re.findall(r"^\[\d+\] \(doc_id=", prompt, re.MULTILINE))
            if ctx_chunks >= 2 and not has_new_rule:
                state["top_k_seen"] = 2

            if "executive salary" in prompt.lower() or "secret bonus" in prompt.lower():
                return json.dumps(
                    {
                        "action": "final",
                        "answer": "The documents do not cover compensation or bonuses.",
                        "citations": [],
                        "refused": True,
                        "refusal_reason": "Out of scope.",
                        "confidence": 0.95,
                    }
                )
            if "hydrophone" in prompt.lower():
                # Regress on top_k == 2 to test rollback!
                if ctx_chunks >= 2 and not has_new_rule:
                    return json.dumps(
                        {
                            "action": "final",
                            "answer": "Cannot determine depth.",
                            "citations": [],
                            "refused": True,
                            "refusal_reason": "Simulated regression",
                            "confidence": 0.2,
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
            # Calibration question: fails at baseline, succeeds once PromptPatch is applied!
            if has_new_rule:
                return json.dumps(
                    {
                        "action": "final",
                        "answer": (
                            "Sensors undergo salt-fog calibration every 90 days at Depot North."
                        ),
                        "citations": [
                            {
                                "doc_id": buoy.doc_id,
                                "chunk_id": buoy.chunk_id,
                                "quote": (
                                    "Sensors undergo salt-fog calibration every 90 days "
                                    "at Depot North."
                                ),
                            }
                        ],
                        "refused": False,
                        "refusal_reason": None,
                        "confidence": 0.95,
                    }
                )
            return json.dumps(
                {
                    "action": "final",
                    "answer": "Sensors undergo periodic calibration.",
                    "citations": [
                        {
                            "doc_id": buoy.doc_id,
                            "chunk_id": buoy.chunk_id,
                            "quote": "Sensors undergo salt-fog calibration",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.6,
                }
            )

        if "Extract all atomic factual claims" in prompt:
            ans = prompt.split("Answer:\n", 1)[1].split("\n\nReturn ONLY", 1)[0].strip()
            return json.dumps({"claims": [ans]})

        if "Verify whether EACH numbered claim" in prompt:
            evidence = prompt.split("Evidence (Retrieved Passages & Tool Outputs):\n", 1)[1].split(
                "\n\nReturn ONLY", 1
            )[0]
            quote = ""
            if "45 meters below mean sea level" in evidence:
                quote = "Deploy hydrophone pods at 45 meters below mean sea level"
            elif "Depot North" in evidence:
                quote = "Sensors undergo salt-fog calibration every 90 days at Depot North."
            return json.dumps(
                {
                    "verdicts": [
                        {"claim_index": 1, "supported": bool(quote), "quote": quote}
                    ]
                }
            )

        if "Determine whether the assistant's answer correctly covers" in prompt:
            return json.dumps({"correct": True, "reason": "ok"})

        raise ValueError(f"Unexpected prompt: {prompt[:120]}")

    import re

    cfg = ForgeConfig()
    llm = FakeLLM(responses=[_sim_responder] * 300, config=cfg)
    run_dir = tmp_path / "opt_run"

    best_def, opt_scorecard = OptimizerLoop(llm, config=cfg, max_iterations=3, bm25_only=True).run(
        defn, eval_set, DOCS_DIR, out_dir=run_dir
    )

    # Iter 1 rolled back (regression), Iter 2 accepted -> 100% dev -> early stop!
    assert opt_scorecard.baseline_dev_score == pytest.approx(2 / 3, abs=0.01)
    assert opt_scorecard.final_dev_score == 1.0
    assert opt_scorecard.iterations_completed == 2
    assert opt_scorecard.iterations[0].rolled_back is True
    assert opt_scorecard.iterations[1].accepted is True
    assert opt_scorecard.stop_reason == "all_dev_cases_passed"
    assert best_def.retrieval.top_k == 1  # rolled back top_k=2!
    expected_rule = "Quote exact numeric values from retrieved excerpts."
    assert expected_rule in best_def.system_prompt.extra_rules

    # Check artifacts & CLI compatibility
    assert (run_dir / "scorecard.json").is_file()
    assert (run_dir / "iterations" / "iter_1" / "log.md").is_file()
    assert (run_dir / "iterations" / "iter_1" / "patch.json").is_file()
    assert (run_dir / "iterations" / "iter_2" / "failures.json").is_file()


def test_cli_forge_optimize_completes_with_early_stop(tmp_path: Path, monkeypatch) -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    _, chunks, profile = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    prov = _prov()

    eval_set = EvalSet(
        agent_name=defn.meta.name,
        eval_writer_model="llama3.1:8b",
        agent_model=defn.model.name,
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[
            EvalCase(
                id="DEV-001",
                category="single_doc",
                split="dev",
                question="What depth should hydrophone pods be deployed?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=acoustic.doc_id, chunk_id=acoustic.chunk_id)
                ],
                key_facts=["Deploy hydrophone pods at 45 meters below mean sea level"],
                provenance=prov,
            )
        ],
        held_out_cases=[],
    )

    run_dir = tmp_path / "run_cli"
    run_dir.mkdir(parents=True)
    defn.save_yaml(run_dir / "definition.yaml")
    (run_dir / "eval_set.json").write_text(eval_set.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    # Always fails dev -> tests 2-consecutive-no-improvement early stop AND 3-iter cap!
    def _no_improve_responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Propose at most 2 targeted patches" in prompt:
            return json.dumps({"patches": []})
        if "User question:" in prompt:
            return json.dumps(
                {
                    "action": "final",
                    "answer": "Unclear depth.",
                    "citations": [
                        {
                            "doc_id": acoustic.doc_id,
                            "chunk_id": acoustic.chunk_id,
                            "quote": "Deploy hydrophone pods",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.4,
                }
            )
        if "Extract all atomic factual claims" in prompt:
            return json.dumps({"claims": ["Unclear depth."]})
        if "Verify whether EACH numbered claim" in prompt:
            return json.dumps(
                {
                    "verdicts": [
                        {"claim_index": 1, "supported": True, "quote": "Deploy hydrophone pods"}
                    ]
                }
            )
        return json.dumps({"correct": False, "reason": "no depth"})

    fake_llm = FakeLLM(responses=[_no_improve_responder] * 100)
    monkeypatch.setattr(
        forge_cli.LLMClient,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: fake_llm),
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["optimize", "--run", str(run_dir), "--max-iters", "3", "--bm25-only"],
    )
    assert res.exit_code == 0
    assert (run_dir / "scorecard.json").is_file()
    sc = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert sc["stop_reason"] == "early_stop_two_consecutive_no_improvement"
    assert sc["iterations_completed"] == 2
