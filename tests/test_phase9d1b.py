"""Tests for Phase 9d-1b: configurable verifier thresholds and HARD canary tier."""

from __future__ import annotations

import json
import re
from pathlib import Path

from forge.config import ForgeConfig, load_config
from forge.evals import EvalCase, EvalProvenance, EvalSet, EvalSourceRef
from forge.ingest import ingest_folder
from forge.judge import run_judge_selftest
from forge.llm import FakeLLM, LLMMessage

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"


def test_judge_thresholds_loaded_from_toml() -> None:
    default_cfg = load_config(profile="default")
    gemini_cfg = load_config(profile="gemini")
    assert default_cfg.judge.claim_support_threshold == 0.72
    assert default_cfg.judge.claim_unsupport_threshold == 0.35
    assert gemini_cfg.judge.claim_support_threshold == 0.72
    assert gemini_cfg.judge.claim_unsupport_threshold == 0.35


def test_hard_canary_tier_separate_reporting_and_all_five_types(tmp_path: Path) -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    chunk_map = {c.chunk_id: c for c in chunks}
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    power = next(c for c in chunks if "14.4 volts" in c.text)
    buoy = next(c for c in chunks if "Depot North" in c.text)

    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)
    eval_set = EvalSet(
        agent_name="sensor-agent",
        eval_writer_model="llama3.1:8b",
        agent_model="qwen2.5:3b-instruct",
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[
            EvalCase(
                id="DEV-001",
                category="multi_doc",
                question="At what depth are pods deployed and what is the lithium voltage?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=acoustic.doc_id, chunk_id=acoustic.chunk_id),
                    EvalSourceRef(doc_id=power.doc_id, chunk_id=power.chunk_id),
                ],
                key_facts=[
                    (
                        "Deploy hydrophone pods at 45 meters below mean sea level "
                        "in calm current zones."
                    ),
                    "The lithium pack delivers 14.4 volts nominal and supports 60 days.",
                ],
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
            EvalCase(
                id="DEV-004",
                category="out_of_scope",
                question="What is the external stock price?",
                expected_behavior="refuse",
                provenance=prov,
            ),
        ],
    )

    def _hard_fake_responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Verify whether EACH numbered claim" in prompt:
            claims_blk = prompt.split("Claims to verify:\n", 1)[1].split("\n\nEvidence", 1)[0]
            ev_blk = prompt.split("Evidence (Retrieved Passages & Tool Outputs):\n", 1)[1].split(
                "\n\nReturn ONLY", 1
            )[0]
            verdicts = []
            for ln in claims_blk.splitlines():
                m = re.match(r"^(\d+)\.\s*(.+)$", ln.strip())
                if not m:
                    continue
                idx = int(m.group(1))
                claim = m.group(2)
                low = claim.lower()
                if "compliance deputy" in low or "reserve buffer" in low:
                    verdicts.append(
                        {
                            "claim_index": idx,
                            "supported": False,
                            "quote": "",
                            "reason": (
                                "The evidence does not state a compliance deputy "
                                "or reserve buffer."
                            ),
                        }
                    )
                elif "edge conditions" in low:
                    verdicts.append(
                        {
                            "claim_index": idx,
                            "supported": True,
                            "quote": acoustic.text.splitlines()[0][:80],
                            "reason": "Non-factual caution consistent with cited evidence.",
                        }
                    )
                else:
                    match_line = next(
                        (
                            el.strip()
                            for el in ev_blk.splitlines()
                            if el.strip() and not el.strip().startswith("[")
                        ),
                        acoustic.text.splitlines()[0],
                    )
                    verdicts.append(
                        {"claim_index": idx, "supported": True, "quote": match_line[:100]}
                    )
            return json.dumps({"verdicts": verdicts})
        if "Determine whether the assistant's answer correctly covers" in prompt:
            return json.dumps({"correct": False, "reason": "Missing second required key fact."})
        return json.dumps({})

    rel = run_judge_selftest(
        FakeLLM(responses=[_hard_fake_responder] * 100),
        chunk_map,
        eval_set=eval_set,
        config=ForgeConfig(),
        out_path=tmp_path / "judge_reliability.json",
    )

    # Standard tier keeps exact 8 standard types only
    assert {m.canary_type for m in rel.per_canary_type} == {
        "correct_refusal",
        "gold_answer",
        "hallucinated_claim",
        "numerically_corrupted",
        "padded_verbose",
        "terse_correct",
        "wrong_document",
        "wrongful_refusal",
    }
    assert rel.total_canaries == 16

    # Hard tier is reported separately with all 5 hard types
    assert {m.canary_type for m in rel.hard_tier_per_canary_type} == {
        "paraphrased_correct",
        "partial_correct_two_facts",
        "subtle_hallucination",
        "wrong_entity_same_shape",
        "correct_with_hedging",
    }
    assert rel.hard_tier_total_canaries >= 9
    assert rel.hard_tier_false_pass_rate == 0.0
    assert rel.hard_tier_false_fail_rate == 0.0
