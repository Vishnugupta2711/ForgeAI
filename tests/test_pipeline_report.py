"""Unit & integration tests for forge.pipeline and forge.report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import AgentDefinition, AgentOutput, AnswerRecord
from forge.cli import app
from forge.config import ForgeConfig
from forge.evals import EvalCase, EvalProvenance, EvalSet
from forge.ingest import ingest_folder
from forge.judge import CaseVerdict, EvalCaseResult, EvalScorecard
from forge.llm import FakeLLM, LLMMessage
from forge.optimizer import OptimizationScorecard
from forge.pipeline import ForgePipeline, PipelineExecutionError
from forge.report import build_report, collect_allowed_numbers, lint_narrative_numbers

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"
HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def _full_pipeline_responder(chunks_by_id: dict[str, str], *, bad_report_number: bool = False):
    """Prompt-routing FakeLLM responder covering every stage of ForgePipeline."""
    counters = {"s": 0, "m": 0}
    cids = list(chunks_by_id.keys())
    acoustic_cid = next((c for c in cids if "45 meters" in chunks_by_id[c]), cids[0])
    buoy_cid = next((c for c in cids if "Depot North" in chunks_by_id[c]), cids[0])

    def _respond(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content

        # Intake stages
        if "Generate ONLY genuinely necessary clarifying questions" in prompt:
            return json.dumps(
                {
                    "questions": [
                        {"id": "Q1", "question": "Preferred citation style?", "why": "format"}
                    ]
                }
            )
        if "simulating a pragmatic requester" in prompt:
            return json.dumps(
                {"answers": [{"question_id": "Q1", "answer": "Use exact chunk quotes."}]}
            )
        if "synthesizing a structured requirements specification" in prompt:
            return json.dumps(
                {
                    "scope": "Hydrophone deployment depth and buoy calibration schedules.",
                    "audience": "Field technicians",
                    "tone": "Concise technical",
                    "must_answer_topics": [
                        {
                            "id": "REQ-001",
                            "title": "Deployment Depth",
                            "description": "Hydrophone pod deployment depth in meters",
                            "priority": "high",
                        }
                    ],
                    "must_refuse_topics": [
                        {
                            "id": "REF-001",
                            "title": "Executive Salaries",
                            "reason": "Not in documentation.",
                        }
                    ],
                    "assumptions": [],
                    "capabilities_needed": [
                        {
                            "name": "tabular_lookup",
                            "reason": "Query sensors.csv inventory",
                            "linked_req_ids": ["REQ-001"],
                            "capability_tag": "csv",
                        }
                    ],
                }
            )

        # Groundability stages
        if "Decompose the brief, requirements, and clarifying Q&A" in prompt:
            return json.dumps(
                {
                    "topics": [
                        {
                            "topic_id": "TOP-001",
                            "title": "Deployment Depth",
                            "description": "Hydrophone pod deployment depth in meters",
                            "source_req_id": "REQ-001",
                            "is_implicit": False,
                        }
                    ]
                }
            )
        if "You are the Groundability Guard." in prompt:
            return json.dumps(
                {
                    "status": "SUPPORTED",
                    "quotes": [
                        {
                            "chunk_id": acoustic_cid,
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "reasoning": "Exact quote found.",
                    "gap_note": None,
                }
            )

        # Generator stages
        if "Propose which tools from the registry" in prompt:
            return json.dumps({"proposed_tools": [{"name": "csv_lookup", "reason": "REQ-001"}]})
        if 'Return ONLY a JSON object:\n{"role":' in prompt:
            return json.dumps({"role": "You are a sensor documentation assistant."})
        if 'Return ONLY a JSON object:\n{"scope":' in prompt:
            return json.dumps({"scope": "Answer hydrophone deployment and calibration questions."})
        if 'Return ONLY a JSON object:\n{"audience_tone":' in prompt:
            return json.dumps({"audience_tone": "Concise technical tone."})
        if 'Return ONLY a JSON object:\n{"grounding_rules":' in prompt:
            return json.dumps({"grounding_rules": "Cite quoted evidence verbatim."})
        if 'Return ONLY a JSON object:\n{"refusal_policy":' in prompt:
            return json.dumps(
                {
                    "refusal_policy": (
                        "Refuse Executive Salaries honestly stating the documents don't cover this."
                    )
                }
            )
        if 'Return ONLY a JSON object:\n{"format_rules":' in prompt:
            return json.dumps({"format_rules": "Follow output contract JSON."})
        if 'Return ONLY a JSON object:\n{"extra_rules":' in prompt:
            return json.dumps({"extra_rules": ["Include units in measurements."]})

        # Eval generator stages
        if "Chunk Reference: doc_id=" in prompt:
            m = re.search(r"chunk_id=([^\s]+)", prompt)
            cid = m.group(1) if m else acoustic_cid
            txt = chunks_by_id[cid]
            line = next((ln.strip() for ln in txt.splitlines() if len(ln.strip()) > 10), txt)
            cnt_m = re.search(r"Generate (\d+) distinct", prompt)
            cnt = int(cnt_m.group(1)) if cnt_m else 2
            items = []
            for _ in range(cnt):
                counters["s"] += 1
                items.append(
                    {
                        "question": f"Single question #{counters['s']} on {cid}: {line[:22]}?",
                        "key_facts": [line[:45]],
                        "notes": f"Tests {cid}",
                    }
                )
            return json.dumps({"cases": items})

        if "REQUIRE combining facts from BOTH chunks below" in prompt:
            m_a = re.search(r"Chunk A: doc_id=[^\s]+ \| chunk_id=([^\s]+)", prompt)
            m_b = re.search(r"Chunk B: doc_id=[^\s]+ \| chunk_id=([^\s]+)", prompt)
            cid_a = m_a.group(1) if m_a else acoustic_cid
            cid_b = m_b.group(1) if m_b else buoy_cid
            line_a = next(
                (ln.strip() for ln in chunks_by_id[cid_a].splitlines() if len(ln.strip()) > 10),
                chunks_by_id[cid_a],
            )
            line_b = next(
                (ln.strip() for ln in chunks_by_id[cid_b].splitlines() if len(ln.strip()) > 10),
                chunks_by_id[cid_b],
            )
            counters["m"] += 1
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": (
                                f"Multi-doc #{counters['m']} combining {cid_a} and {cid_b}?"
                            ),
                            "key_facts": [line_a[:40], line_b[:40]],
                            "notes": "Cross-doc",
                        }
                    ]
                }
            )

        if "out-of-scope evaluation questions" in prompt:
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": f"OOS executive salary question #{i}?",
                            "subtype": "must_refuse",
                            "notes": "Refuse OOS",
                        }
                        for i in range(1, 7)
                    ]
                }
            )

        if "edge-case evaluation questions" in prompt:
            line = next(
                (
                    ln.strip()
                    for ln in chunks_by_id[acoustic_cid].splitlines()
                    if len(ln.strip()) > 10
                ),
                chunks_by_id[acoustic_cid],
            )
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": f"Paraphrase variant #{i}: {line[:22]}?",
                            "expected_behavior": "answer",
                            "source_chunk_ids": [acoustic_cid],
                            "key_facts": [line[:40]],
                            "subtype": "typo_paraphrase",
                            "notes": "Paraphrase",
                        }
                        for i in range(1, 5)
                    ]
                }
            )

        if "specifically require invoking the attached tools" in prompt:
            csv_cid = next((c for c in cids if "sensors.csv" in c), acoustic_cid)
            line = chunks_by_id[csv_cid].splitlines()[0]
            return json.dumps(
                {
                    "cases": [
                        {
                            "question": f"Tool lookup #{i} for {line}?",
                            "target_tool": "csv_lookup",
                            "source_chunk_ids": [csv_cid],
                            "key_facts": [line],
                            "notes": "Tool lookup",
                        }
                        for i in range(1, 5)
                    ]
                }
            )

        if "Determine whether the question below can be answered using ONLY" in prompt:
            return json.dumps({"solvable": True, "reason": "Solvable from chunk."})
        if "Determine whether the candidate out-of-scope question IS actually answerable" in prompt:
            return json.dumps({"answerable_from_docs": False, "reason": "Not in docs."})

        # Query rewrite
        if "Rewrite the user question into a concise" in prompt:
            return json.dumps({"rewritten_query": "hydrophone deployment depth calibration"})

        # AgentRuntime answers
        if "User question:" in prompt:
            low_p = prompt.lower()
            if "oos" in low_p or "salary" in low_p or "confidential" in low_p:
                return json.dumps(
                    {
                        "action": "final",
                        "answer": "The documents don't cover executive salaries.",
                        "citations": [],
                        "refused": True,
                        "refusal_reason": "Out of scope.",
                        "confidence": 0.95,
                    }
                )
            # Extract matching chunk from prompt so citations always match retrieved context
            chunk_pat = r"\[\d+\] \(doc_id=([^ |]+) \| chunk_id=([^\s|]+)[^\n]*\n(.+?)(?:\n\n|\Z)"
            m_ret = re.search(chunk_pat, prompt, re.DOTALL)
            if "45 meters" in prompt and "hydrophone" in low_p:
                for m in re.finditer(chunk_pat, prompt, re.DOTALL):
                    if "45 meters" in m.group(3):
                        m_ret = m
                        break
            if m_ret:
                doc_id, chunk_id = m_ret.group(1), m_ret.group(2)
                chunk_lines = [ln.strip() for ln in m_ret.group(3).splitlines() if ln.strip()]
                quote_text = next(
                    (ln for ln in chunk_lines if "45 meters" in ln),
                    chunk_lines[-1] if chunk_lines else m_ret.group(3).strip(),
                )
                answer_text = " ".join(chunk_lines)
            else:
                doc_id, chunk_id = "acoustic_guide.md", acoustic_cid
                quote_text = "Deploy hydrophone pods at 45 meters below mean sea level"
                answer_text = quote_text
            return json.dumps(
                {
                    "action": "final",
                    "answer": f"Based on the documentation: {answer_text}",
                    "citations": [
                        {"doc_id": doc_id, "chunk_id": chunk_id, "quote": quote_text[:80]}
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.9,
                }
            )

        # Judge stages
        if "Extract all atomic factual claims" in prompt:
            ans = prompt.split("Answer:\n", 1)[1].split("\n\nReturn ONLY", 1)[0].strip()
            return json.dumps({"claims": [ans]})
        if "Verify whether EACH numbered claim" in prompt:
            ev = prompt.split("Evidence (Retrieved Passages & Tool Outputs):\n", 1)[1].split(
                "\n\nReturn ONLY", 1
            )[0]
            first_ev_line = next(
                (
                    ln.strip()
                    for ln in ev.splitlines()
                    if ln.strip() and not ln.strip().startswith("[")
                ),
                "",
            )
            if "override code" in prompt.lower():
                return json.dumps(
                    {"verdicts": [{"claim_index": 1, "supported": False, "quote": ""}]}
                )
            return json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_index": 1,
                            "supported": bool(first_ev_line),
                            "quote": first_ev_line[:80],
                        }
                    ]
                }
            )
        if "Determine whether the assistant's answer correctly covers" in prompt:
            return json.dumps({"correct": True, "reason": "Matches key facts."})

        # Optimizer
        if "Propose at most 2 targeted patches" in prompt:
            return json.dumps(
                {
                    "patches": [
                        {
                            "kind": "prompt",
                            "section": "extra_rules",
                            "op": "add_rule",
                            "text": "Quote exact measurements directly from retrieved chunks.",
                            "addressed_case_ids": [],
                            "rationale": "Improve factual precision.",
                        }
                    ]
                }
            )

        # Report diagnosis
        if "What Still Fails and Why" in prompt:
            if bad_report_number:
                return json.dumps(
                    {
                        "diagnosis_items": [
                            {
                                "case_ids": ["HELD-001"],
                                "diagnosis": "HELD-001 failed with 99.4% fabricated error rate.",
                            }
                        ],
                        "overall_summary": "HELD-001 failed (99.4%).",
                    }
                )
            return json.dumps(
                {
                    "diagnosis_items": [
                        {
                            "case_ids": ["HELD-001"],
                            "diagnosis": "HELD-001 failed due to missed key facts.",
                        }
                    ],
                    "overall_summary": "HELD-001 remains unresolved.",
                }
            )

        raise ValueError(f"Unhandled prompt in pipeline test: {prompt[:120]}")

    return _respond


def test_narrative_lint_rejects_invented_number_and_missing_ids(tmp_path: Path) -> None:
    allowed = collect_allowed_numbers({"score": 0.75, "passed": 3, "total": 4})
    assert "75" in allowed
    assert "3" in allowed

    ok, errs = lint_narrative_numbers(
        "Case HELD-001 failed; overall 3 of 4 (75.0%) passed.",
        allowed,
        valid_case_ids={"HELD-001"},
    )
    assert ok and not errs

    bad_ok, bad_errs = lint_narrative_numbers(
        "Case HELD-001 failed with an invented 88.8% score.",
        allowed,
        valid_case_ids={"HELD-001"},
    )
    assert bad_ok is False
    assert any("88.8" in e for e in bad_errs)

    # Full report rejects LLM diagnosis containing invented number and falls back deterministically
    run_dir = tmp_path / "lint_run"
    (run_dir / "final_held_out_eval").mkdir(parents=True)
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.save_yaml(run_dir / "definition.yaml")
    held_sc = EvalScorecard(
        split="held_out",
        agent_name=defn.meta.name,
        agent_model="qwen2.5:3b-instruct",
        judge_model="llama3.1:8b",
        same_model_as_agent=False,
        same_family_as_agent=False,
        total_cases=2,
        passed_cases=1,
        overall_pass_rate=0.5,
        overall_wilson_low=0.0945,
        overall_wilson_high=0.9055,
        results=[
            EvalCaseResult(
                case_id="HELD-001",
                category="single_doc",
                split="held_out",
                question="Q?",
                expected_behavior="answer",
                answer_record=AnswerRecord(
                    question="Q?",
                    output=AgentOutput(answer="Wrong", citations=[], refused=False),
                ),
                verdict=CaseVerdict(
                    case_id="HELD-001",
                    category="single_doc",
                    expected_behavior="answer",
                    grounded=False,
                    correct=False,
                    refusal_ok=True,
                    passed=False,
                    reasons=["Missing citations"],
                ),
            )
        ],
    )
    held_sc.save(run_dir / "final_held_out_eval")

    bad_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "diagnosis_items": [
                        {
                            "case_ids": ["HELD-001"],
                            "diagnosis": "HELD-001 suffered a 99.4% retrieval failure.",
                        }
                    ],
                    "overall_summary": "HELD-001 failed 99.4% of checks.",
                }
            )
        ]
    )
    report_path, _ = build_report(run_dir, llm_client=bad_llm)
    report_text = report_path.read_text(encoding="utf-8")
    assert "Narrative Number Lint Note" in report_text
    assert "99.4" not in report_text.split("## 6. What Still Fails and Why")[1].split(
        "Narrative Number Lint Note"
    )[0]
    assert "HELD-001" in report_text


def test_report_suspicion_rule_fires_on_100_or_large_gap(tmp_path: Path) -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)

    # 1. Suspicion fires on large dev vs held-out gap
    run_gap = tmp_path / "gap_run"
    run_gap.mkdir()
    defn.save_yaml(run_gap / "definition.yaml")
    prov = EvalProvenance(prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42)
    eval_set = EvalSet(
        agent_name=defn.meta.name,
        eval_writer_model="llama3.1:8b",
        agent_model=defn.model.name,
        same_model_as_agent=False,
        same_family_as_agent=False,
        seed=42,
        dev_cases=[],
        held_out_cases=[
            EvalCase(
                id="HELD-001",
                category="out_of_scope",
                split="held_out",
                question="Q?",
                expected_behavior="refuse",
                provenance=prov,
            )
        ],
    )
    eval_set.save(run_gap)
    opt_gap = OptimizationScorecard(
        agent_name=defn.meta.name,
        baseline_dev_score=0.60,
        final_dev_score=0.90,
        baseline_held_out_score=0.50,
        final_held_out_score=0.50,
        iterations_completed=1,
        stop_reason="max_iterations",
    )
    opt_gap.save(run_gap / "scorecard.json")

    rep_gap, csv_gap = build_report(run_gap)
    text_gap = rep_gap.read_text(encoding="utf-8")
    assert "Suspicion Rule Triggered" in text_gap
    assert "Large dev vs held-out gap" in text_gap
    assert csv_gap.is_file()

    # 2. Suspicion fires on 100% dev/held-out score
    run_100 = tmp_path / "run_100"
    run_100.mkdir()
    defn.save_yaml(run_100 / "definition.yaml")
    eval_set.save(run_100)
    opt_100 = opt_gap.model_copy(
        update={"final_dev_score": 1.0, "final_held_out_score": 1.0}
    )
    opt_100.save(run_100 / "scorecard.json")
    rep_100, _ = build_report(run_100)
    assert "100.0%" in rep_100.read_text(encoding="utf-8")
    assert "Suspicion Rule Triggered" in rep_100.read_text(encoding="utf-8")


def test_full_pipeline_fast_crash_partial_report_and_resume(tmp_path: Path) -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    responder = _full_pipeline_responder({c.chunk_id: c.text for c in chunks})
    cfg = ForgeConfig()
    run_dir = tmp_path / "runs" / "demo"

    # 1. Simulate mid-pipeline crash right at 'evals' stage
    llm_crash = FakeLLM(responses=[responder] * 300, config=cfg)

    def _boom() -> None:
        raise RuntimeError("Simulated mid-pipeline power loss")

    progress_1: list[tuple[str, str]] = []
    pipe1 = ForgePipeline(
        llm_crash,
        config=cfg,
        bm25_only=True,
        progress_cb=lambda st, msg, _el: progress_1.append((st, msg)),
    )
    with pytest.raises(PipelineExecutionError) as exc_info:
        pipe1.run(
            brief="Answer hydrophone deployment and sensor inventory questions.",
            docs_dir=DOCS_DIR,
            out_dir=run_dir,
            mode="auto",
            fast=True,
            stage_hooks={"evals": _boom},
        )
    assert exc_info.value.stage == "evals"
    assert (run_dir / "report.md").is_file()
    partial_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Partial Run Warning" in partial_md
    assert "Simulated mid-pipeline power loss" in partial_md

    # 2. Resume from checkpoint: completed stages skipped, remaining stages finish!
    llm_resume = FakeLLM(
        responses=[responder] * 400,
        config=cfg,
        usage_log_path=run_dir / "usage.jsonl",
    )
    progress_2: list[tuple[str, str]] = []
    pipe2 = ForgePipeline(
        llm_resume,
        config=cfg,
        bm25_only=True,
        progress_cb=lambda st, msg, _el: progress_2.append((st, msg)),
    )
    final_state = pipe2.run(
        brief="Answer hydrophone deployment and sensor inventory questions.",
        docs_dir=DOCS_DIR,
        out_dir=run_dir,
        mode="auto",
        fast=True,
        resume=True,
    )

    assert ("ingest", "skipped (resumed)") in progress_2
    assert ("intake", "skipped (resumed)") in progress_2
    assert ("groundability", "skipped (resumed)") in progress_2
    assert ("generate", "skipped (resumed)") in progress_2
    assert set(final_state.completed_stages) >= {
        "ingest",
        "intake",
        "groundability",
        "generate",
        "evals",
        "judge_selftest",
        "optimize",
        "report",
    }
    assert (run_dir / "definition.yaml").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "scorecard.csv").is_file()
    assert (run_dir / "judge_reliability.json").is_file()

    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Run Summary",
        "## 2. Requirements & Groundability Flags",
        "## 3. Final Agent Definition & Tool Gate Decisions",
        "## 4. Per-Iteration Dev Scorecard & Held-Out Results",
        "## 5. Failure-Category Counts per Iteration",
        "## 6. What Still Fails and Why",
        "## 7. Judge Reliability & Bias Checks",
        "## 8. Limitations",
    ):
        assert heading in report_text

    # Confirm final definition.yaml loads cleanly in AgentDefinition
    loaded_def = AgentDefinition.load_yaml(run_dir / "definition.yaml")
    assert loaded_def.meta.name


def test_cli_forge_run_and_chat_acceptance(tmp_path: Path, monkeypatch) -> None:
    """Acceptance test: `forge run` produces report.md + definition.yaml usable by `forge chat`."""
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    responder = _full_pipeline_responder({c.chunk_id: c.text for c in chunks})
    fake_llm = FakeLLM(responses=[responder] * 500, config=ForgeConfig())

    monkeypatch.setattr(
        forge_cli.LLMClient,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: fake_llm),
    )

    brief_file = tmp_path / "brief.txt"
    brief_file.write_text(
        "Answer hydrophone deployment depth and sensor inventory questions.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "runs" / "acceptance"
    runner = CliRunner()
    res_run = runner.invoke(
        app,
        [
            "run",
            "--brief",
            str(brief_file),
            "--docs",
            str(DOCS_DIR),
            "--out",
            str(out_dir),
            "--auto",
            "--fast",
            "--bm25-only",
        ],
    )
    assert res_run.exit_code == 0
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "scorecard.csv").is_file()
    assert (out_dir / "definition.yaml").is_file()

    # Run `forge chat` against the produced definition.yaml
    res_chat = runner.invoke(
        app,
        [
            "chat",
            str(out_dir / "definition.yaml"),
            "--docs",
            str(DOCS_DIR),
            "--question",
            "What depth should hydrophone pods be deployed?",
            "--bm25-only",
        ],
    )
    assert res_chat.exit_code == 0
    assert "ANSWERED" in res_chat.stdout
    plain_stdout = " ".join(re.sub(r"[│╭╮╰╯─]", " ", res_chat.stdout).split())
    assert "45 meters" in plain_stdout
    assert "acoustic_guide.md" in plain_stdout
