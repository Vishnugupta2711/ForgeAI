"""Unit tests for forge.generator (prompt sections, heuristics, tool gate, smoke, cache)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.agent import AgentDefinition
from forge.config import ForgeConfig
from forge.generator import AgentGenerator, derive_retrieval_settings, gate_tools
from forge.groundability import GroundabilityReport, TopicGroundability
from forge.ingest import CorpusProfile, CsvSchemaProfile, DocumentProfile, ingest_folder
from forge.intake import (
    CapabilityNeed,
    RefusalTopic,
    RequirementsDocument,
    RequirementTopic,
)
from forge.llm import FakeLLM
from forge.retrieval import HybridIndex
from forge.tools import default_registry

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"


def _make_profile(
    *,
    doc_count: int = 3,
    avg_chars: int = 1200,
    headings_per_doc: int = 3,
    tabular_count: int = 0,
) -> CorpusProfile:
    docs: list[DocumentProfile] = []
    for i in range(doc_count):
        is_tab = i < tabular_count
        docs.append(
            DocumentProfile(
                doc_id=f"doc_{i}.{'csv' if is_tab else 'md'}",
                name=f"doc_{i}.{'csv' if is_tab else 'md'}",
                path=f"/tmp/doc_{i}",
                doc_type="csv" if is_tab else "md",
                size_bytes=avg_chars,
                char_count=avg_chars,
                line_count=40,
                page_count=1,
                chunk_count=max(2, avg_chars // 400),
                headings=[f"Section {h}" for h in range(headings_per_doc)],
                csv_profile=(
                    CsvSchemaProfile(
                        columns=["id", "value"],
                        dtypes={"id": "str", "value": "int"},
                        sample_rows=[{"id": "1", "value": "10"}],
                        row_count=5,
                    )
                    if is_tab
                    else None
                ),
            )
        )
    return CorpusProfile(
        root_path="/tmp/docs",
        total_files_scanned=doc_count,
        loaded_docs=doc_count,
        skipped_files=0,
        total_chunks=sum(d.chunk_count for d in docs),
        documents=docs,
    )


def _make_req(
    *,
    brief: str,
    topics: list[RequirementTopic] | None = None,
    refusals: list[RefusalTopic] | None = None,
    capabilities: list[CapabilityNeed] | None = None,
    gaps: list[str] | None = None,
) -> RequirementsDocument:
    report = (
        GroundabilityReport(
            has_gaps=True,
            unsupported_count=len(gaps),
            flagged_gaps=gaps,
            topics=[
                TopicGroundability(
                    topic_id=f"TOP-{i + 1}",
                    title=g,
                    description=g,
                    status="UNSUPPORTED",
                    coverage_score=0.0,
                    key_term_overlap=0.0,
                    max_retrieval_score=0.0,
                )
                for i, g in enumerate(gaps)
            ],
        )
        if gaps
        else GroundabilityReport(has_gaps=False, supported_count=1)
    )
    return RequirementsDocument(
        brief=brief,
        scope="Answer documented operational questions.",
        audience="Operations team",
        tone="Concise and factual",
        must_answer_topics=topics
        or [
            RequirementTopic(
                id="REQ-001",
                title="Core Guidance",
                description="Answer documented procedural guidelines.",
            )
        ],
        must_refuse_topics=refusals or [],
        capabilities_needed=capabilities or [],
        groundability_report=report,
    )


@pytest.mark.parametrize(
    ("case_name", "brief", "req_topics", "capabilities", "tabular_count", "proposed", "expected"),
    [
        (
            "policy_qa_only",
            "Answer employee handbook policy rules from the PDF docs.",
            [RequirementTopic(id="REQ-001", title="Policy Rules", description="Handbook rules")],
            [],
            0,
            ["csv_lookup", "http_api"],
            [],
        ),
        (
            "text_only_never_gets_http_or_csv",
            "Summarize maintenance rules from plain text manuals.",
            [RequirementTopic(id="REQ-001", title="Maintenance", description="Manual rules")],
            [
                CapabilityNeed(
                    name="tabular_lookup",
                    reason="Maybe look up data",
                    linked_req_ids=["REQ-001"],
                ),
                CapabilityNeed(
                    name="live_external_data",
                    reason="Maybe check external site",
                    linked_req_ids=["REQ-001"],
                ),
            ],
            0,
            ["csv_lookup", "http_api"],
            [],
        ),
        (
            "catalogue_lookup_plus_totals",
            "Look up inventory parts in the catalogue CSV and compute order totals.",
            [
                RequirementTopic(
                    id="REQ-001",
                    title="Part Lookup and Cost Totals",
                    description="Filter parts CSV and calculate total order cost",
                )
            ],
            [
                CapabilityNeed(
                    name="tabular_lookup",
                    reason="Filter catalogue CSV rows",
                    linked_req_ids=["REQ-001"],
                ),
                CapabilityNeed(
                    name="arithmetic",
                    reason="Compute total order prices",
                    linked_req_ids=["REQ-001"],
                ),
            ],
            1,
            ["csv_lookup", "calculator", "http_api"],
            ["csv_lookup", "calculator"],
        ),
        (
            "csv_lookup_without_arithmetic",
            "Find matching sensor entries in sensors.csv by model name.",
            [RequirementTopic(id="REQ-001", title="Sensor Lookup", description="Find rows")],
            [
                CapabilityNeed(
                    name="tabular_lookup",
                    reason="Filter sensor CSV",
                    linked_req_ids=["REQ-001"],
                )
            ],
            1,
            ["csv_lookup", "calculator"],
            ["csv_lookup"],
        ),
        (
            "explicit_weather_need",
            "Provide coastal safety guidance and check live weather forecast via Open-Meteo.",
            [
                RequirementTopic(
                    id="REQ-001",
                    title="Weather Forecast",
                    description="Fetch live weather forecast for deployment coordinates",
                )
            ],
            [
                CapabilityNeed(
                    name="live_external_data",
                    reason="Fetch live weather forecast",
                    linked_req_ids=["REQ-001"],
                )
            ],
            0,
            ["http_api"],
            ["http_api"],
        ),
        (
            "current_date_need",
            "Check equipment calibration expiration relative to today's date.",
            [
                RequirementTopic(
                    id="REQ-001",
                    title="Calibration Expiry",
                    description="Compare calibration dates against current date",
                )
            ],
            [
                CapabilityNeed(
                    name="current_date",
                    reason="Know today's date to check expiration",
                    linked_req_ids=["REQ-001"],
                )
            ],
            0,
            ["current_date"],
            ["current_date"],
        ),
        (
            "unlinked_or_dropped_req_id_rejected",
            "Calculate totals from CSV.",
            [RequirementTopic(id="REQ-001", title="Text Rules", description="Plain guidance")],
            [
                CapabilityNeed(
                    name="tabular_lookup",
                    reason="Query CSV",
                    linked_req_ids=["REQ-999"],
                ),
                CapabilityNeed(
                    name="arithmetic",
                    reason="Compute totals",
                    linked_req_ids=[],
                ),
            ],
            1,
            ["csv_lookup", "calculator"],
            [],
        ),
        (
            "full_multi_capability_brief",
            (
                "Query parts CSV, compute unit cost totals, check today's date for lead times, "
                "and check live weather forecast at destination port."
            ),
            [
                RequirementTopic(
                    id="REQ-001",
                    title="CSV Totals & Lead Time",
                    description="Filter parts, calculate cost, compare against current date",
                ),
                RequirementTopic(
                    id="REQ-002",
                    title="Port Weather",
                    description="Fetch live weather forecast for arrival port",
                ),
            ],
            [
                CapabilityNeed(name="tabular_lookup", reason="CSV", linked_req_ids=["REQ-001"]),
                CapabilityNeed(name="arithmetic", reason="Totals", linked_req_ids=["REQ-001"]),
                CapabilityNeed(name="current_date", reason="Today", linked_req_ids=["REQ-001"]),
                CapabilityNeed(
                    name="live_external_data", reason="Weather", linked_req_ids=["REQ-002"]
                ),
            ],
            1,
            ["csv_lookup", "calculator", "current_date", "http_api"],
            ["csv_lookup", "calculator", "http_api", "current_date"],
        ),
    ],
)
def test_tool_selection_matrix(
    case_name: str,
    brief: str,
    req_topics: list[RequirementTopic],
    capabilities: list[CapabilityNeed],
    tabular_count: int,
    proposed: list[str],
    expected: list[str],
) -> None:
    profile = _make_profile(doc_count=2, tabular_count=tabular_count)
    req = _make_req(brief=brief, topics=req_topics, capabilities=capabilities)
    registry = default_registry()

    attached, decisions = gate_tools(req, profile, registry, llm_proposed=proposed)
    assert set([t.name for t in attached]) == set(expected), f"Failed matrix case: {case_name}"
    assert len(decisions) == 4
    for spec in attached:
        assert spec.justification_requirement_id
    for dec in decisions:
        assert dec.reason


def test_retrieval_heuristics_vary_with_corpus_profile() -> None:
    req = _make_req(brief="Answer technical questions.")
    small_prose = _make_profile(doc_count=2, avg_chars=600, headings_per_doc=3, tabular_count=0)
    large_tabular = _make_profile(
        doc_count=8, avg_chars=2400, headings_per_doc=2, tabular_count=6
    )

    s1, r1 = derive_retrieval_settings(small_prose, req)
    s2, r2 = derive_retrieval_settings(large_tabular, req)

    assert s1.top_k < s2.top_k
    assert s1.hybrid_alpha > s2.hybrid_alpha  # tabular shifts alpha toward BM25
    assert s1.query_rewrite is False
    assert s2.query_rewrite is True
    assert r1["inputs"]["loaded_docs"] == 2
    assert r2["inputs"]["tabular_share"] == 0.75


def _scripted_generation_and_smoke_responses() -> list[str]:
    return [
        # 1. tool proposal
        json.dumps({"proposed_tools": [{"name": "csv_lookup", "reason": "REQ-001"}]}),
        # 2-8. 7 prompt sections (note grounding_rules omits quote intentionally to test guard!)
        json.dumps({"role": "You are a sensor documentation assistant."}),
        json.dumps({"scope": "Answer acoustic array and sensor table questions."}),
        json.dumps({"audience_tone": "Concise engineering tone."}),
        json.dumps({"grounding_rules": "Only state verified facts."}),
        json.dumps({"refusal_policy": "Refuse unsupported queries."}),
        json.dumps({"format_rules": "Follow output contract JSON strictly."}),
        json.dumps({"extra_rules": ["Use csv_lookup for row queries."]}),
        # 9-14. 3 smoke test questions (each has query_rewrite=True on mixed 4-doc corpus + final)
        json.dumps({"rewritten_query": "hydrophone deployment depth"}),
        json.dumps(
            {
                "answer": "Deploy hydrophone pods at 45 meters.",
                "citations": [
                    {
                        "doc_id": "acoustic_guide.md",
                        "chunk_id": "acoustic_guide.md::c2",
                        "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                    }
                ],
                "refused": False,
                "refusal_reason": None,
                "confidence": 0.9,
            }
        ),
        json.dumps({"rewritten_query": "sensor inventory specs sensors.csv"}),
        json.dumps(
            {
                "answer": "SNS-101 is HydroPod-A with 60 battery days.",
                "citations": [
                    {
                        "doc_id": "sensors.csv",
                        "chunk_id": "sensors.csv::row-1",
                        "quote": "sensor_id: SNS-101",
                    }
                ],
                "refused": False,
                "refusal_reason": None,
                "confidence": 0.88,
            }
        ),
        json.dumps({"rewritten_query": "vendor pricing"}),
        json.dumps(
            {
                "answer": "The documents don't cover internal pricing.",
                "citations": [],
                "refused": True,
                "refusal_reason": "Pricing is outside documented scope.",
                "confidence": 0.95,
            }
        ),
    ]


def test_generator_end_to_end_and_determinism_with_cache(tmp_path: Path) -> None:
    ingest_res, chunks, profile = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    index = HybridIndex.build(chunks, llm_client=None)

    req = _make_req(
        brief="Answer hydrophone deployment and sensor CSV questions.",
        topics=[
            RequirementTopic(
                id="REQ-001",
                title="Deployment Depth",
                description="Hydrophone pod depth",
            ),
            RequirementTopic(
                id="REQ-002",
                title="Sensor Inventory",
                description="Query sensor specs in sensors.csv",
            ),
        ],
        refusals=[
            RefusalTopic(
                id="REF-001",
                title="Vendor Pricing",
                reason="Not present in documents",
                from_gap=True,
            )
        ],
        capabilities=[
            CapabilityNeed(
                name="tabular_lookup",
                reason="Query sensors.csv",
                linked_req_ids=["REQ-002"],
            )
        ],
        gaps=["[UNSUPPORTED] Vendor Pricing: Missing from corpus."],
    )

    cache_dir = tmp_path / "llm_cache"
    cfg = ForgeConfig()
    cfg.generation.temperature = 0.0
    cfg.generation.seed = 42

    llm1 = FakeLLM(
        responses=_scripted_generation_and_smoke_responses(),
        config=cfg,
        cache_dir=cache_dir,
    )
    out1 = tmp_path / "run1"
    defn1, log1 = AgentGenerator(llm1, config=cfg).generate(
        req,
        profile,
        index,
        documents=ingest_res.documents,
        out_dir=out1,
        run_smoke_test=True,
    )

    assert (out1 / "definition.yaml").is_file()
    assert (out1 / "generation_log.json").is_file()
    assert defn1.attached_tool_names() == ["csv_lookup"]
    assert "quote" in defn1.system_prompt.grounding_rules.lower()
    assert "doc_id" in defn1.system_prompt.format_rules.lower()
    assert "chunk_id" in defn1.system_prompt.format_rules.lower()
    assert "quote" in defn1.system_prompt.format_rules.lower()
    assert "Vendor Pricing" in defn1.system_prompt.refusal_policy
    assert "don't cover" in defn1.system_prompt.refusal_policy.lower()
    assert len(log1.smoke_test_results) == 3
    assert all(r.ok for r in log1.smoke_test_results)
    assert len(defn1.meta.provenance.prompt_hashes) == 8

    # Reload from YAML and verify identical schema
    reloaded = AgentDefinition.load_yaml(out1 / "definition.yaml")
    assert reloaded.meta.name == defn1.meta.name

    # Second run with empty FakeLLM sharing the same cache_dir proves temperature=0 determinism!
    llm2 = FakeLLM(responses=[], config=cfg, cache_dir=cache_dir)
    defn2, log2 = AgentGenerator(llm2, config=cfg).generate(
        req,
        profile,
        index,
        documents=ingest_res.documents,
        run_smoke_test=True,
    )
    assert len(llm2.fake_provider.chat_calls) == 0
    assert defn2.system_prompt == defn1.system_prompt
    assert defn2.retrieval == defn1.retrieval
    assert defn2.tools == defn1.tools
    assert log2.llm_usage_summary["cache_hits"] == 14
