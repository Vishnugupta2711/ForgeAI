"""Unit tests for forge.intake and forge.groundability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forge.groundability import GroundabilityGuard, StrictGroundabilityError
from forge.ingest import ingest_folder
from forge.intake import IntakeInterviewer, RequirementsDocument, RequirementTopic
from forge.llm import FakeLLM
from forge.retrieval import HybridIndex

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"


def test_intake_question_cap_and_auto_assumptions_recorded() -> None:
    _, _, profile = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)

    # LLM returns 8 questions; code must truncate strictly to 5
    eight_questions = {
        "questions": [
            {"id": f"Q{i}", "question": f"Clarifying question {i}?", "why": "disambiguate"}
            for i in range(1, 9)
        ]
    }
    auto_answers = {
        "answers": [
            {"question_id": f"Q{i}", "answer": f"Auto-derived answer {i}."}
            for i in range(1, 6)
        ]
    }
    synth_req = {
        "scope": "Answer marine sensor deployment and calibration questions.",
        "audience": "Field technicians",
        "tone": "Concise technical",
        "must_answer_topics": [
            {
                "id": "REQ-001",
                "title": "Deployment Depth",
                "description": "Depth and placement of hydrophone pods",
                "priority": "high",
            }
        ],
        "must_refuse_topics": [],
        "assumptions": ["Synthesized baseline assumption."],
        "capabilities_needed": [
            {
                "name": "tabular_lookup",
                "reason": "Query sensor inventory CSV",
                "linked_req_ids": ["REQ-001"],
                "capability_tag": "csv",
            }
        ],
    }

    llm = FakeLLM(
        responses=[
            json.dumps(eight_questions),
            json.dumps(auto_answers),
            json.dumps(synth_req),
        ]
    )
    interviewer = IntakeInterviewer(llm)
    req = interviewer.run("Build a sensor docs assistant.", profile, mode="auto")

    assert len(req.clarifying_qa) == 5
    assert [qa.question_id for qa in req.clarifying_qa] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert all(qa.source == "auto" for qa in req.clarifying_qa)
    for i in range(1, 6):
        assert any(f"Auto-assumed (Q{i}:" in a for a in req.assumptions)


def test_intake_interactive_and_answers_file_modes(tmp_path: Path) -> None:
    _, _, profile = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)

    q_payload = json.dumps(
        {
            "questions": [
                {"id": "Q1", "question": "Target audience?", "why": "tone"},
                {"id": "Q2", "question": "Include CSV lookups?", "why": "tools"},
            ]
        }
    )
    synth_payload = json.dumps(
        {
            "scope": "Technical support over sensor manuals.",
            "audience": "Field engineers",
            "tone": "Direct",
            "must_answer_topics": [
                {
                    "id": "REQ-001",
                    "title": "Calibration Schedule",
                    "description": "Salt-fog calibration intervals",
                    "priority": "high",
                }
            ],
            "must_refuse_topics": [],
            "assumptions": [],
            "capabilities_needed": [],
        }
    )

    # 1. Interactive mode
    llm_interactive = FakeLLM(responses=[q_payload, synth_payload])
    answers_iter = iter(["Field engineers", "Yes"])
    req_inter = IntakeInterviewer(llm_interactive).run(
        "Answer sensor docs.",
        profile,
        mode="interactive",
        input_fn=lambda _: next(answers_iter),
    )
    assert [qa.answer for qa in req_inter.clarifying_qa] == ["Field engineers", "Yes"]
    assert all(qa.source == "interactive" for qa in req_inter.clarifying_qa)

    # 2. --answers YAML file mode
    answers_file = tmp_path / "answers.yaml"
    answers_file.write_text(
        yaml.safe_dump({"Q1": "Operations crew", "Q2": "Yes, filter sensors.csv"}),
        encoding="utf-8",
    )
    llm_file = FakeLLM(responses=[q_payload, synth_payload])
    req_file = IntakeInterviewer(llm_file).run(
        "Answer sensor docs.",
        profile,
        mode="answers",
        answers_path=answers_file,
    )
    assert req_file.clarifying_qa[0].answer == "Operations crew"
    assert req_file.clarifying_qa[0].source == "answers_file"


def test_groundability_flags_absent_topic_common_knowledge_and_fabricated_quote(
    tmp_path: Path,
) -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    index = HybridIndex.build(chunks, llm_client=None)

    req = RequirementsDocument(
        brief=(
            "Answer hydrophone deployment depth, European capital cities (common knowledge), "
            "employee vacation reimbursement policy (absent), and battery power budget."
        ),
        scope="Answer technical and general questions.",
        audience="Operators",
        tone="Concise",
        must_answer_topics=[
            RequirementTopic(
                id="REQ-001",
                title="Deployment Depth",
                description="Hydrophone pod deployment depth in meters",
            ),
            RequirementTopic(
                id="REQ-002",
                title="European Capitals",
                description="Capital cities of France and Germany from general world knowledge",
            ),
            RequirementTopic(
                id="REQ-003",
                title="Vacation Reimbursement",
                description="Employee annual vacation travel reimbursement limits",
            ),
            RequirementTopic(
                id="REQ-004",
                title="Power Budget",
                description="Lithium battery nominal voltage and winter solar charging",
            ),
        ],
    )

    decomposed = {
        "topics": [
            {
                "topic_id": "TOP-001",
                "title": "Deployment Depth",
                "description": "Hydrophone pod deployment depth in meters",
                "source_req_id": "REQ-001",
                "is_implicit": False,
            },
            {
                "topic_id": "TOP-002",
                "title": "European Capitals",
                "description": "Capital cities of France and Germany",
                "source_req_id": "REQ-002",
                "is_implicit": False,
            },
            {
                "topic_id": "TOP-003",
                "title": "Vacation Reimbursement",
                "description": "Employee annual vacation travel reimbursement limits",
                "source_req_id": "REQ-003",
                "is_implicit": False,
            },
            {
                "topic_id": "TOP-004",
                "title": "Power Budget",
                "description": "Lithium battery voltage and winter solar charging",
                "source_req_id": "REQ-004",
                "is_implicit": False,
            },
        ]
    }

    # TOP-001: valid verbatim quote -> SUPPORTED
    verdict_top1 = {
        "status": "SUPPORTED",
        "quotes": [
            {
                "chunk_id": "acoustic_guide.md::c2",
                "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
            }
        ],
        "reasoning": "Exact deployment depth is documented.",
        "gap_note": None,
    }
    # TOP-002 (common knowledge) & TOP-003 (absent): deterministic overlap=0.0 catches them
    # before trusting any LLM response.
    # TOP-004: FakeLLM claims SUPPORTED with a FABRICATED quote (not in chunk) -> caught!
    verdict_top4_fabricated = {
        "status": "SUPPORTED",
        "quotes": [
            {
                "chunk_id": "acoustic_guide.md::c3",
                "quote": "The lithium pack delivers 24 volts and lasts 365 days.",
            }
        ],
        "reasoning": "Fabricated quote test.",
        "gap_note": None,
    }

    llm = FakeLLM(
        responses=[
            json.dumps(decomposed),
            json.dumps(verdict_top1),
            json.dumps(verdict_top4_fabricated),
        ]
    )
    guard = GroundabilityGuard(index, llm, top_k=4, alpha=0.0)
    updated, report = guard.evaluate(req, strict=False)

    assert report.supported_count == 1
    assert report.unsupported_count == 3
    assert report.has_gaps is True

    by_topic = {t.topic_id: t for t in report.topics}
    assert by_topic["TOP-001"].status == "SUPPORTED"
    assert by_topic["TOP-001"].quotes[0].verified_substring is True

    # Absent & common-knowledge topics must NOT be marked supported
    assert by_topic["TOP-002"].status == "UNSUPPORTED"
    assert by_topic["TOP-003"].status == "UNSUPPORTED"

    # Fabricated quote must be caught and downgraded to UNSUPPORTED
    assert by_topic["TOP-004"].status == "UNSUPPORTED"
    assert by_topic["TOP-004"].quote_verification_failed is True

    # Unsupported topics moved to must_refuse and removed from must_answer
    assert [t.id for t in updated.must_answer_topics] == ["REQ-001"]
    refusal_titles = {r.title for r in updated.must_refuse_topics}
    assert {"European Capitals", "Vacation Reimbursement", "Power Budget"}.issubset(
        refusal_titles
    )

    _, md_path = updated.save(tmp_path)
    md_text = md_path.read_text(encoding="utf-8")
    assert "Groundability Gaps Flagged" in md_text
    assert "## Groundability Report" in md_text


def test_groundability_partial_narrows_scope_and_strict_aborts() -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    index = HybridIndex.build(chunks, llm_client=None)

    req = RequirementsDocument(
        brief="Answer calibration schedule and replacement vendor pricing.",
        scope="Answer calibration and vendor pricing questions.",
        audience="Technicians",
        tone="Direct",
        must_answer_topics=[
            RequirementTopic(
                id="REQ-001",
                title="Calibration Schedule",
                description="Calibration frequency and depot location",
            )
        ],
    )

    decomposed = {
        "topics": [
            {
                "topic_id": "TOP-001",
                "title": "Calibration Schedule",
                "description": "Calibration frequency and depot location",
                "source_req_id": "REQ-001",
                "is_implicit": False,
            }
        ]
    }
    partial_verdict = {
        "status": "PARTIAL",
        "quotes": [
            {
                "chunk_id": "buoy_manual.txt::c2",
                "quote": "Sensors undergo salt-fog calibration every 90 days at Depot North.",
            }
        ],
        "reasoning": "Schedule and depot are documented, pricing is missing.",
        "gap_note": "Vendor pricing is not covered.",
    }

    llm = FakeLLM(responses=[json.dumps(decomposed), json.dumps(partial_verdict)])
    guard = GroundabilityGuard(index, llm, top_k=3, alpha=0.0)
    updated, report = guard.evaluate(req, strict=False)

    assert report.partial_count == 1
    assert "Scope narrowed due to partial corpus evidence" in updated.scope
    assert updated.must_answer_topics[0].groundability_status == "PARTIAL"

    # --strict mode raises StrictGroundabilityError with attached requirements
    llm_strict = FakeLLM(responses=[json.dumps(decomposed), json.dumps(partial_verdict)])
    guard_strict = GroundabilityGuard(index, llm_strict, top_k=3, alpha=0.0)
    with pytest.raises(StrictGroundabilityError) as exc_info:
        guard_strict.evaluate(req, strict=True)
    assert exc_info.value.report.has_gaps is True
