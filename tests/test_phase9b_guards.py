"""Tests for Phase 9b guards: min_score, precheck, operator map, matcher, resume, provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import AgentDefinition, AgentRuntime
from forge.cli import app
from forge.config import ForgeConfig, load_prompt
from forge.evals import EvalCase, EvalGenerator, EvalProvenance, EvalSet, EvalSourceRef
from forge.ingest import ingest_folder
from forge.judge.deterministic import compute_key_fact_recall, match_key_fact
from forge.llm import FakeLLM, LLMMessage
from forge.optimizer import (
    ClassifiedFailure,
    OptimizerLoop,
    PromptPatch,
    RetrievalPatch,
    validate_patch_operator_mapping,
    validate_retrieval_patch_scale,
)
from forge.pipeline import ForgePipeline, PipelineExecutionError, PipelineResumeError
from forge.report import build_report
from forge.retrieval import HybridIndex, HybridRetriever, RetrievalConfig
from forge.tools import ToolContext

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures"
DOCS_DIR = FIXTURES_ROOT / "docs"
HAND_AGENT_PATH = FIXTURES_ROOT / "hand_agent.yaml"


def _prov(model: str = "llama3.1:8b", seed: int = 42) -> EvalProvenance:
    return EvalProvenance(prompt_file="p.md", prompt_hash="h", model=model, seed=seed)


# ---------------------------------------------------------------------------
# Item 1: min_score normalized scale & runtime gate
# ---------------------------------------------------------------------------


def test_retriever_score_scale_and_runtime_min_score_0_1_keeps_chunks() -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    llm = FakeLLM()
    index = HybridIndex.build(chunks, llm_client=llm)

    retriever = HybridRetriever(
        index, llm_client=llm, config=RetrievalConfig(alpha=0.5, top_k=4, min_score=0.1)
    )
    scale = retriever.score_scale()
    assert scale["scale"] == "normalized_0_to_1"
    assert scale["raw_max"] == pytest.approx(1.0 / 61.0, abs=1e-5)
    assert scale["normalized_min"] == 0.0
    assert scale["normalized_max"] == 1.0

    # With raw RRF ~0.015, min_score=0.1 normalized must NOT empty retrieval
    hits = retriever.search("hydrophone deployment depth")
    assert len(hits) >= 1
    assert all(0.0 < h.score < 0.05 for h in hits)
    assert all(retriever.normalize_score(h.score) >= 0.1 for h in hits)

    # Runtime min_score=0.1 uses same normalized scale and keeps chunks across queries
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.retrieval.min_score = 0.1
    defn.retrieval.query_rewrite = False
    acoustic = next(c for c in chunks if "45 meters" in c.text)
    runtime_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "action": "final",
                    "answer": "Deploy at 45 meters.",
                    "citations": [
                        {
                            "doc_id": acoustic.doc_id,
                            "chunk_id": acoustic.chunk_id,
                            "quote": "Deploy hydrophone pods at 45 meters",
                        }
                    ],
                    "refused": False,
                    "confidence": 0.9,
                }
            )
            for _ in range(3)
        ]
    )
    tool_ctx = ToolContext.from_documents([], offline_only=True)
    runtime = AgentRuntime(defn, index, llm_client=runtime_llm, tool_context=tool_ctx)
    for query in (
        "Where do we deploy hydrophone pods?",
        "How often are sensors calibrated?",
        "Which depot handles sensors?",
    ):
        rec = runtime.answer(query)
        assert len(rec.retrieved_chunks) >= 1
        assert any("min_score gate (0.1 normalized)" in t for t in rec.trace)

    # Patch scale validation rejects impossible bounds and clamps excessive safe threshold
    bad_patch = RetrievalPatch(target_fields={"min_score": 1.5}, addressed_case_ids=["DEV-001"])
    kept, err = validate_retrieval_patch_scale(bad_patch, scale)
    assert kept is None and err is not None

    high_patch = RetrievalPatch(target_fields={"min_score": 0.92}, addressed_case_ids=["DEV-001"])
    clamped, err2 = validate_retrieval_patch_scale(high_patch, scale)
    assert clamped is not None and err2 is None
    assert clamped.target_fields["min_score"] == 0.7


# ---------------------------------------------------------------------------
# Item 2: Patch precheck gate + 1 alternative operator within same iteration
# ---------------------------------------------------------------------------


def test_precheck_rejects_empty_retrieval_and_schema_failure_and_tries_alternative(
    tmp_path: Path,
) -> None:
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
                expected_sources=[
                    EvalSourceRef(doc_id=buoy.doc_id, chunk_id=buoy.chunk_id)
                ],
                key_facts=["Sensors undergo salt-fog calibration every 90 days at Depot North."],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-003",
                category="single_doc",
                split="dev",
                question="qqqqwwww?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=buoy.doc_id, chunk_id=buoy.chunk_id)
                ],
                key_facts=["Sensors undergo salt-fog calibration every 90 days at Depot North."],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-004",
                category="single_doc",
                split="dev",
                question="xxxyyyzzz?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=buoy.doc_id, chunk_id=buoy.chunk_id)
                ],
                key_facts=["Sensors undergo salt-fog calibration every 90 days at Depot North."],
                provenance=prov,
            ),
            EvalCase(
                id="DEV-005",
                category="single_doc",
                split="dev",
                question="mmmnnnppp?",
                expected_behavior="answer",
                expected_sources=[
                    EvalSourceRef(doc_id=buoy.doc_id, chunk_id=buoy.chunk_id)
                ],
                key_facts=["Sensors undergo salt-fog calibration every 90 days at Depot North."],
                provenance=prov,
            ),
        ],
        held_out_cases=[],
    )

    judge_calls_seen = {"count": 0}

    def _responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Propose at most 2 targeted patches" in prompt:
            # First propose a RetrievalPatch with min_score=0.70 that empties 2/3 unmatched
            # queries -> precheck catches it, skips full eval, and runs alternative PromptPatch!
            return json.dumps(
                {
                    "patches": [
                        {
                            "kind": "retrieval",
                            "target_fields": {"min_score": 0.70},
                            "addressed_case_ids": ["DEV-003"],
                            "rationale": "Overly strict floor test.",
                        }
                    ]
                }
            )
        if "User question:" in prompt:
            sys_prompt = messages[0].content
            if "Include all relevant facts and exact units" in sys_prompt:
                # Alternative operator fixes all 4 cases cleanly!
                if "hydrophone" in prompt.lower():
                    return json.dumps(
                        {
                            "action": "final",
                            "answer": "Deploy hydrophone pods at 45 meters below mean sea level.",
                            "citations": [
                                {
                                    "doc_id": acoustic.doc_id,
                                    "chunk_id": acoustic.chunk_id,
                                    "quote": (
                                        "Deploy hydrophone pods at 45 meters below mean sea level"
                                    ),
                                }
                            ],
                            "refused": False,
                            "confidence": 0.95,
                        }
                    )
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
                        "confidence": 0.95,
                    }
                )
            # Baseline fails DEV-003..005 (wrong chunk / missed fact)
            if any(tok in prompt.lower() for tok in ("qqqqwwww", "xxxyyyzzz", "mmmnnnppp")):
                return json.dumps(
                    {
                        "action": "final",
                        "answer": "Sensors are active.",
                        "citations": [
                            {
                                "doc_id": buoy.doc_id,
                                "chunk_id": buoy.chunk_id,
                                "quote": "Sensors undergo salt-fog calibration",
                            }
                        ],
                        "refused": False,
                        "confidence": 0.5,
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
                    "confidence": 0.9,
                }
            )
        if "Extract all atomic factual claims" in prompt:
            judge_calls_seen["count"] += 1
            ans = prompt.split("Answer:\n", 1)[1].split("\n\nReturn ONLY", 1)[0].strip()
            return json.dumps({"claims": [ans]})
        if "Verify whether EACH numbered claim" in prompt:
            ev = prompt.split("Evidence (Retrieved Passages & Tool Outputs):\n", 1)[1].split(
                "\n\nReturn ONLY", 1
            )[0]
            q = (
                "Deploy hydrophone pods at 45 meters below mean sea level"
                if "45 meters" in ev
                else "Sensors undergo salt-fog calibration every 90 days at Depot North."
            )
            return json.dumps({"verdicts": [{"claim_index": 1, "supported": True, "quote": q}]})
        return json.dumps({"correct": True, "reason": "ok"})

    cfg = ForgeConfig()
    llm = FakeLLM(responses=[_responder] * 300, config=cfg)
    opt = OptimizerLoop(llm, config=cfg, max_iterations=2, bm25_only=False)
    # Give DEV-003 a nonexistent gold chunk so classifier emits retrieval_miss for RetrievalPatch
    eval_set.dev_cases[2].expected_sources = [
        EvalSourceRef(doc_id="missing.txt", chunk_id="missing.txt::c99")
    ]
    best_def, scorecard = opt.run(defn, eval_set, DOCS_DIR, out_dir=tmp_path / "precheck_run")

    assert scorecard.iterations_completed >= 1
    it1 = scorecard.iterations[0]
    assert it1.rejected_precheck
    assert "rejected_precheck" in it1.rejected_precheck[0]
    # Alternative operator (RetrievalPatch top_k=3) ran within the same iteration!
    assert isinstance(it1.proposed_patches[0], RetrievalPatch)
    assert best_def.retrieval.min_score is None
    assert "Precheck Rejections" in (tmp_path / "precheck_run/iterations/iter_1/log.md").read_text()

    # Schema failure precheck case: patch on format_rules triggers broken JSON -> skipped!
    bad_schema_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "patches": [
                        {
                            "kind": "prompt",
                            "section": "format_rules",
                            "op": "add_rule",
                            "text": "Broken rule probe.",
                            "addressed_case_ids": ["DEV-001"],
                            "rationale": "Test schema probe failure.",
                        }
                    ]
                }
            ),
            # 3 probe calls x (1 initial + 2 repairs) = 9 broken JSON outputs -> schema fail!
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
            "NOT JSON",
        ]
    )
    probe_opt = OptimizerLoop(bad_schema_llm, config=cfg, max_iterations=1, bm25_only=True)
    patched_def = probe_opt._prepare_candidate(
        defn,
        [PromptPatch(section="format_rules", op="add_rule", text="Broken rule probe.")],
        DOCS_DIR,
        [],
        HybridIndex.build(chunks, llm_client=None),
        {c.chunk_id: c for c in chunks},
    )[0]
    ok_schema, reason_schema = probe_opt._precheck_candidate(
        patched_def,
        [PromptPatch(section="format_rules", op="add_rule", text="Broken rule probe.")],
        eval_set.dev_cases[:3],
        HybridIndex.build(chunks, llm_client=None),
        [],
    )
    assert ok_schema is False
    assert reason_schema is not None and "contract schema validation failed" in reason_schema


# ---------------------------------------------------------------------------
# Item 3: Operator mapping audit
# ---------------------------------------------------------------------------


def test_operator_mapping_blocks_wrong_patch_and_requires_grounding() -> None:
    failures = [
        ClassifiedFailure(
            case_id="DEV-014",
            eval_category="single_doc",
            failure_category="prompt_defect",
            subtype="context_ignored_or_incomplete",
            evidence="Right context retrieved",
        ),
        ClassifiedFailure(
            case_id="DEV-015",
            eval_category="single_doc",
            failure_category="hallucination",
            subtype="ungrounded_content",
            evidence="Ungrounded number",
        ),
        ClassifiedFailure(
            case_id="DEV-016",
            eval_category="single_doc",
            failure_category="prompt_defect",
            subtype="contract_failure",
            evidence="Schema validation failed",
        ),
    ]

    bad_ret = RetrievalPatch(
        target_fields={"top_k": 8, "min_score": 0.1},
        addressed_case_ids=["DEV-014"],
        rationale="Wrongly targeted retrieval at prompt_defect",
    )
    bad_hall = PromptPatch(
        section="extra_rules",
        op="add_rule",
        text="Verify facts.",
        addressed_case_ids=["DEV-015"],
        rationale="Wrong section for hallucination",
    )
    good_contract = PromptPatch(
        section="format_rules",
        op="add_rule",
        text="Emit valid JSON object with required citation keys.",
        addressed_case_ids=["DEV-016"],
        rationale="Fix contract_failure",
    )
    good_hall = PromptPatch(
        section="grounding_rules",
        op="add_rule",
        text="Never output unsupported numbers.",
        addressed_case_ids=["DEV-015"],
        rationale="Fix hallucination with grounding_rules",
    )

    valid, rejected = validate_patch_operator_mapping(
        [bad_ret, bad_hall, good_contract, good_hall], failures
    )
    assert len(rejected) == 2
    assert any("no retrieval_miss" in r for r in rejected)
    assert any("grounding_rules" in r for r in rejected)
    assert valid == [good_contract, good_hall]


# ---------------------------------------------------------------------------
# Item 4: Key-fact matcher snake_case / abbreviation normalization + numeric guard
# ---------------------------------------------------------------------------


def test_matcher_normalizes_kv_abbreviations_while_rejecting_wrong_numbers() -> None:
    assert match_key_fact(
        "min_order_qty: 10",
        "The minimum order quantity required for the Zinc Anode Block MK-5 is 10 units.",
    )
    assert match_key_fact(
        "min_order_qty: 4",
        "The minimum order quantity required is 4 units across all suppliers.",
    )
    assert match_key_fact("MOQ 12", "The minimum order quantity is 12 units.")
    assert match_key_fact("unit_price_usd: 67.00", "The unit price is $67 per piece.")

    # Numerically corrupted values MUST still fail
    assert not match_key_fact(
        "min_order_qty: 10",
        "The minimum order quantity required for the Zinc Anode Block MK-5 is 12 units.",
    )
    assert not match_key_fact("unit_price_usd: 67.00", "The unit price is $76 per piece.")


def test_real_ship_spares_kv_matcher_regression_both_directions() -> None:
    """Exact DEV-014/016/019/020 cases: real answers pass; altered numbers fail."""
    cases = [
        (
            "DEV-014",
            "min_order_qty: 10",
            (
                "The minimum order quantity required for the "
                "Anti-Corrosion Zinc Anode Block MK-5 is 10 units."
            ),
            (
                "The minimum order quantity required for the "
                "Anti-Corrosion Zinc Anode Block MK-5 is 12 units."
            ),
        ),
        (
            "DEV-016",
            "min_order_qty: 4",
            (
                "The minimum order quantity required for the Boiler Burner Igniter "
                "Assembly MK-4 is 4 units across all available supplier listings "
                "in the catalogue."
            ),
            (
                "The minimum order quantity required for the Boiler Burner Igniter "
                "Assembly MK-4 is 5 units across all available supplier listings "
                "in the catalogue."
            ),
        ),
        (
            "DEV-019",
            "min_order_qty: 2",
            (
                "The minimum order quantity required for the Refrigerant R404A Service "
                "Cylinder MK-2 is 2 units across all listed suppliers and part numbers "
                "in the catalogue."
            ),
            (
                "The minimum order quantity required for the Refrigerant R404A Service "
                "Cylinder MK-2 is 3 units across all listed suppliers and part numbers "
                "in the catalogue."
            ),
        ),
        (
            "DEV-020",
            "min_order_qty: 2",
            (
                "The minimum order quantity required for the Main Engine Fuel Injector "
                "MK-2 is 2 units across all listed suppliers and locations in the "
                "spares catalogue."
            ),
            (
                "The minimum order quantity required for the Main Engine Fuel Injector "
                "MK-2 is 3 units across all listed suppliers and locations in the "
                "spares catalogue."
            ),
        ),
    ]
    for cid, key_fact, real_answer, corrupted_answer in cases:
        assert match_key_fact(key_fact, real_answer), f"{cid} real answer failed"
        recall_ok, _, _ = compute_key_fact_recall([key_fact], real_answer)
        assert recall_ok == 1.0, f"{cid} recall expected 1.0"

        assert not match_key_fact(key_fact, corrupted_answer), (
            f"{cid} corrupted number unexpectedly matched"
        )
        recall_bad, _, _ = compute_key_fact_recall([key_fact], corrupted_answer)
        assert recall_bad == 0.0, f"{cid} corrupted recall expected 0.0"


# ---------------------------------------------------------------------------
# Item 5 & 7: Resume safety, --allow-model-change, usage.jsonl headers, warnings
# ---------------------------------------------------------------------------


def test_resume_refuses_model_change_allows_with_flag_and_builds_usage_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    acoustic = next(c for c in chunks if "45 meters" in c.text)

    def _responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "clarifying questions" in prompt.lower():
            return json.dumps({"questions": []})
        if "answer the clarification questions" in prompt.lower():
            return json.dumps({"answers": []})
        if (
            "synthesizing final agent requirements" in prompt.lower()
            or ("scope" in prompt.lower() and "must_answer" in prompt.lower())
        ):
            return json.dumps(
                {
                    "scope": "Answer hydrophone deployment and sensor calibration questions.",
                    "audience": "Engineers",
                    "tone": "Technical",
                    "must_answer_topics": [
                        {
                            "id": "REQ-001",
                            "title": "Hydrophone Depth",
                            "description": "Depth guidance",
                        }
                    ],
                    "must_refuse_topics": [
                        {"id": "REF-001", "title": "Payroll", "reason": "Out of scope"}
                    ],
                    "capabilities_needed": [],
                    "assumptions": [],
                }
            )
        if "atomic topics" in prompt.lower():
            return json.dumps(
                {
                    "topics": [
                        {
                            "topic_id": "TOP-001",
                            "title": "Hydrophone Depth",
                            "description": "Depth guidance",
                            "source_req_id": "REQ-001",
                        }
                    ]
                }
            )
        if "evidence verifier" in prompt.lower():
            return json.dumps(
                {
                    "status": "SUPPORTED",
                    "quotes": [
                        {
                            "chunk_id": acoustic.chunk_id,
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "reasoning": "Supported",
                }
            )
        if "system prompt field" in prompt.lower():
            for field, val in {
                '"role"': {"role": "Assistant."},
                '"scope"': {"scope": "Hydrophone docs only."},
                '"audience_tone"': {"audience_tone": "Technical."},
                '"grounding_rules"': {"grounding_rules": "Cite exact chunks."},
                '"refusal_policy"': {"refusal_policy": "Refuse OOS asks."},
                '"format_rules"': {"format_rules": "JSON with doc_id, chunk_id, quote."},
                '"extra_rules"': {"extra_rules": ["Be concise."]},
            }.items():
                if field in prompt:
                    return json.dumps(val)
            return json.dumps({"proposed_tools": []})
        if "single_doc" in prompt.lower() or "chunk excerpt" in prompt.lower():
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": "What depth should pods be deployed?",
                            "key_facts": ["45 meters"],
                            "notes": "",
                        }
                    ]
                }
            )
        if "multi_doc" in prompt.lower() or "two document" in prompt.lower():
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": "Where are pods deployed and calibrated?",
                            "key_facts": ["45 meters", "Depot North"],
                            "notes": "",
                        }
                    ]
                }
            )
        if "out_of_scope" in prompt.lower() or "out-of-scope" in prompt.lower():
            return json.dumps(
                {"cases": [{"question": "What is the CEO salary?", "subtype": "oos", "notes": ""}]}
            )
        if "edge" in prompt.lower():
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": "Depth for hydrophone?",
                            "expected_behavior": "answer",
                            "source_chunk_ids": [acoustic.chunk_id],
                            "key_facts": ["45 meters"],
                            "subtype": "short",
                            "notes": "",
                        }
                    ]
                }
            )
        if "solvable" in prompt.lower():
            return json.dumps({"solvable": True, "reason": "ok"})
        if "unanswerable" in prompt.lower() or "answerable_from_docs" in prompt.lower():
            return json.dumps({"answerable_from_docs": False, "reason": "oos"})
        if "Extract all atomic factual claims" in prompt:
            return json.dumps({"claims": ["Deploy at 45 meters."]})
        if "Verify whether EACH numbered claim" in prompt:
            return json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_index": 1,
                            "supported": True,
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ]
                }
            )
        if "Determine whether the assistant's answer" in prompt:
            return json.dumps({"correct": True, "reason": "ok"})
        if "Propose at most 2 targeted patches" in prompt:
            return json.dumps({"patches": []})
        if "What Still Fails and Why" in prompt:
            return json.dumps({"diagnosis_items": [], "overall_summary": "All ok."})
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
                "confidence": 0.9,
            }
        )

    run_dir = tmp_path / "resume_guard_run"
    cfg_a = ForgeConfig()
    cfg_a.models.agent = "qwen2.5:3b-instruct"
    cfg_a.models.judge = "llama3.1:8b"
    cfg_a.models.eval_writer = "llama3.1:8b"

    llm_a = FakeLLM(
        responses=[_responder] * 200,
        config=cfg_a,
        usage_log_path=run_dir / "usage.jsonl",
    )
    with pytest.raises(PipelineExecutionError):
        ForgePipeline(llm_a, config=cfg_a, bm25_only=True).run(
            brief="Hydrophone guide assistant.",
            docs_dir=DOCS_DIR,
            out_dir=run_dir,
            mode="auto",
            fast=True,
            stage_hooks={"evals": lambda: (_ for _ in ()).throw(RuntimeError("stop_here"))},
        )

    # Now attempt --resume with changed judge model WITHOUT --allow-model-change -> must REFUSE
    cfg_b = cfg_a.model_copy(deep=True)
    cfg_b.models.judge = "qwen2.5:7b-instruct"
    llm_b = FakeLLM(
        responses=[_responder] * 300,
        config=cfg_b,
        usage_log_path=run_dir / "usage.jsonl",
    )

    with pytest.raises(PipelineResumeError) as ref_exc:
        ForgePipeline(llm_b, config=cfg_b, bm25_only=True).run(
            brief="Hydrophone guide assistant.",
            docs_dir=DOCS_DIR,
            out_dir=run_dir,
            mode="auto",
            fast=True,
            resume=True,
            allow_model_change=False,
        )
    msg = str(ref_exc.value)
    assert "judge" in msg
    assert "llama3.1:8b" in msg
    assert "qwen2.5:7b-instruct" in msg
    assert "groundability" in msg

    # CLI must also refuse with non-zero exit code when FORGE_JUDGE_MODEL changes config
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setattr(
        forge_cli.LLMClient, "from_config", classmethod(lambda cls, *a, **kw: llm_b)
    )
    cli_res = CliRunner().invoke(
        app,
        [
            "run",
            "--brief",
            "Hydrophone guide assistant.",
            "--docs",
            str(DOCS_DIR),
            "--out",
            str(run_dir),
            "--auto",
            "--fast",
            "--resume",
            "--bm25-only",
        ],
    )
    assert cli_res.exit_code == 1
    assert "Resume Refused" in cli_res.stdout

    # Now allow model change -> invalidates groundability+downstream (keeps ingest & intake!)
    progress: list[tuple[str, str]] = []
    final_state = ForgePipeline(
        llm_b,
        config=cfg_b,
        bm25_only=True,
        progress_cb=lambda s, m, _e: progress.append((s, m)),
    ).run(
        brief="Hydrophone guide assistant.",
        docs_dir=DOCS_DIR,
        out_dir=run_dir,
        mode="auto",
        fast=True,
        resume=True,
        allow_model_change=True,
    )
    assert ("ingest", "skipped (resumed)") in progress
    assert ("intake", "skipped (resumed)") in progress
    assert ("groundability", "running") in progress
    assert final_state.model_change_history

    rep_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "usage.jsonl actuals" in rep_text
    assert "Model Changes During Resume" in rep_text
    # Same-Model-Family & Same-Provider warnings both fire for qwen2.5:3b vs 7b on ollama
    assert "Same-Model-Family Warning" in rep_text
    assert "Same-Provider Warning" in rep_text

    # Also test report.md header prefers usage.jsonl actuals over stale state metadata
    stale_dir = tmp_path / "stale_header_test"
    stale_dir.mkdir()
    (stale_dir / "pipeline_state.json").write_text(
        json.dumps(
            {
                "profile": "ollama",
                "providers": {"agent": "ollama", "judge": "ollama"},
                "models": {"agent": "stale-agent-1b", "judge": "stale-judge-1b"},
                "completed_stages": ["ingest"],
                "failed_stage": "optimize",
                "error": "quota",
            }
        ),
        encoding="utf-8",
    )
    (stale_dir / "usage.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "role": "agent",
                        "provider": "gemini",
                        "model": "actual-agent-v2",
                        "status": "ok",
                        "total_tokens": 10,
                    }
                ),
                json.dumps(
                    {
                        "role": "judge",
                        "provider": "gemini",
                        "model": "actual-judge-v2",
                        "status": "ok",
                        "total_tokens": 15,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rep_stale, _ = build_report(stale_dir)
    stale_txt = rep_stale.read_text(encoding="utf-8")
    assert "actual-agent-v2" in stale_txt
    assert "actual-judge-v2" in stale_txt
    assert "stale-judge-1b" not in stale_txt
    assert "Same-Model-Family Warning" in stale_txt
    assert "Same-Provider Warning" in stale_txt


# ---------------------------------------------------------------------------
# Item 6: Per-case provenance in eval_set.json
# ---------------------------------------------------------------------------


def test_eval_set_populates_per_case_and_top_level_provenance(tmp_path: Path) -> None:
    _, chunks, profile = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    acoustic = next(c for c in chunks if "45 meters" in c.text)

    def _eval_responder(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if (
            "Return JSON" in prompt
            and "single_doc" in prompt.lower()
            or "chunk excerpt" in prompt.lower()
        ):
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": "At what depth are hydrophone pods deployed?",
                            "key_facts": [
                                "Deploy hydrophone pods at 45 meters below mean sea level"
                            ],
                            "notes": "s1",
                        }
                    ]
                }
            )
        if "two document" in prompt.lower():
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": (
                                "Which depot handles salt-fog calibration and at what pod depth?"
                            ),
                            "key_facts": ["Depot North", "45 meters"],
                            "notes": "m1",
                        }
                    ]
                }
            )
        if "out-of-scope" in prompt.lower():
            return json.dumps(
                {
                    "cases": [
                        {"question": "What is executive bonus?", "subtype": "oos", "notes": "o1"}
                    ]
                }
            )
        if "edge" in prompt.lower():
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": "Hydrophone depth?",
                            "expected_behavior": "answer",
                            "source_chunk_ids": [acoustic.chunk_id],
                            "key_facts": ["45 meters"],
                            "subtype": "short",
                            "notes": "e1",
                        }
                    ]
                }
            )
        if "solvable" in prompt.lower():
            return json.dumps({"solvable": True, "reason": "ok"})
        return json.dumps({"answerable_from_docs": False, "reason": "oos"})

    cfg = ForgeConfig()
    cfg.models.eval_writer = "llama3.1:8b"
    cfg.generation.seed = 77
    llm = FakeLLM(responses=[_eval_responder] * 100, config=cfg)
    index = HybridIndex.build(chunks, llm_client=llm)
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    from forge.intake.models import RequirementsDocument

    req = RequirementsDocument(
        brief="Hydrophone guide",
        scope="Hydrophone deployment and buoy calibration",
        audience="Engineers",
        tone="Clear",
    )
    EvalGenerator(llm, config=cfg).generate(
        defn,
        req,
        profile,
        index,
        quotas={"single_doc": 1, "multi_doc": 1, "out_of_scope": 1, "edge_case": 1, "tool_use": 0},
        seed=77,
        out_dir=tmp_path / "eval_out",
    )
    persisted = json.loads((tmp_path / "eval_out" / "eval_set.json").read_text(encoding="utf-8"))
    assert persisted["provenance"]["eval_writer_model"] == "llama3.1:8b"
    assert persisted["provenance"]["seed"] == 77
    assert persisted["provenance"]["prompt_hashes"]

    all_cases = [*persisted["dev_cases"], *persisted["held_out_cases"]]
    assert len(all_cases) >= 2
    for item in all_cases:
        prov = item["provenance"]
        assert prov["prompt_file"].startswith("evalgen_")
        expected_hash = hashlib.sha256(
            load_prompt(prov["prompt_file"]).encode("utf-8")
        ).hexdigest()[:16]
        assert prov["prompt_hash"] == expected_hash
        assert prov["model"] == "llama3.1:8b"
        assert prov["seed"] >= 77
