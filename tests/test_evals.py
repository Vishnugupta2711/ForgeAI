"""Unit tests for forge.evals (quality gates, quotas, stratified split, FakeLLM)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import AgentDefinition
from forge.cli import app
from forge.config import ForgeConfig
from forge.evals import (
    EvalCase,
    EvalGenerator,
    EvalProvenance,
    EvalSourceRef,
    stratified_split,
    verify_case_key_facts,
)
from forge.ingest import ingest_folder
from forge.intake import RequirementsDocument, RequirementTopic
from forge.llm import FakeLLM, LLMMessage
from forge.retrieval import HybridIndex

DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"
HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def _sample_requirements() -> RequirementsDocument:
    return RequirementsDocument(
        brief="Answer hydrophone deployment, buoy telemetry, safety rules, and sensor CSV queries.",
        scope="Marine acoustic array, buoy maintenance, and sensor inventory.",
        audience="Field engineers",
        tone="Concise technical",
        must_answer_topics=[
            RequirementTopic(
                id="REQ-001",
                title="Deployment Depth",
                description="Hydrophone pod depth and shipping lane clearance",
            ),
            RequirementTopic(
                id="REQ-002",
                title="Calibration & Telemetry",
                description="Buoy packet interval and calibration schedule",
            ),
        ],
    )


def _smart_evalgen_responder(chunks_by_id: dict[str, str]):
    """Deterministic prompt-inspecting FakeLLM callback producing rich eval responses."""
    counter = {"s": 0, "m": 0}

    def _respond(messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        if "Determine whether the question below can be answered using ONLY" in prompt:
            if "UNSOLVABLE_MARKER" in prompt:
                return json.dumps({"solvable": False, "reason": "Missing info"})
            return json.dumps({"solvable": True, "reason": "Supported by gold passage"})

        if "Determine whether the candidate out-of-scope question IS actually answerable" in prompt:
            if "ANSWERABLE_LEAK_MARKER" in prompt:
                return json.dumps(
                    {"answerable_from_docs": True, "reason": "Found directly in docs"}
                )
            return json.dumps({"answerable_from_docs": False, "reason": "Not in docs"})

        if "Chunk Reference: doc_id=" in prompt:
            m = re.search(r"chunk_id=([^\s]+)", prompt)
            cid = m.group(1) if m else next(iter(chunks_by_id))
            text = chunks_by_id[cid]
            first_line = next(
                (ln.strip() for ln in text.splitlines() if len(ln.strip()) > 12), text
            )
            cases = []
            # Request count parsed from prompt
            cnt_m = re.search(r"Generate (\d+) distinct", prompt)
            cnt = int(cnt_m.group(1)) if cnt_m else 2
            for _ in range(cnt):
                counter["s"] += 1
                idx = counter["s"]
                # Inject one fabricated key fact on idx==2 to test Gate 1 dropping!
                facts = (
                    ["Totally fabricated nonexistent fact XYZ 999"]
                    if idx == 2
                    else [first_line[:45]]
                )
                cases.append(
                    {
                        "question": f"Single-doc question #{idx} about {cid} ({first_line[:24]})?",
                        "key_facts": facts,
                        "notes": f"Tests {cid}",
                    }
                )
            return json.dumps({"cases": cases})

        if "REQUIRE combining facts from BOTH chunks below" in prompt:
            m_a = re.search(r"Chunk A: doc_id=[^\s]+ \| chunk_id=([^\s]+)", prompt)
            m_b = re.search(r"Chunk B: doc_id=[^\s]+ \| chunk_id=([^\s]+)", prompt)
            cid_a = m_a.group(1) if m_a else next(iter(chunks_by_id))
            cid_b = m_b.group(1) if m_b else next(iter(chunks_by_id))
            line_a = next(
                (ln.strip() for ln in chunks_by_id[cid_a].splitlines() if len(ln.strip()) > 10),
                chunks_by_id[cid_a],
            )
            line_b = next(
                (ln.strip() for ln in chunks_by_id[cid_b].splitlines() if len(ln.strip()) > 10),
                chunks_by_id[cid_b],
            )
            cnt_m = re.search(r"Generate (\d+) multi-document", prompt)
            cnt = int(cnt_m.group(1)) if cnt_m else 2
            cases = []
            for _ in range(cnt):
                counter["m"] += 1
                idx = counter["m"]
                cases.append(
                    {
                        "question": f"Multi-doc question #{idx} combining {cid_a} and {cid_b}?",
                        "key_facts": [line_a[:40], line_b[:40]],
                        "notes": "Cross-doc synthesis",
                    }
                )
            return json.dumps({"cases": cases})

        if "out-of-scope evaluation questions" in prompt:
            subtypes = [
                "must_refuse",
                "groundability_gap",
                "general_knowledge",
                "prompt_injection",
            ]
            cases = [
                {
                    "question": "ANSWERABLE_LEAK_MARKER: What depth are hydrophones deployed?",
                    "subtype": "general_knowledge",
                    "notes": "Should be rejected by Gate 3",
                }
            ]
            for i in range(1, 12):
                st = subtypes[i % len(subtypes)]
                cases.append(
                    {
                        "question": (
                            f"OOS {st} question #{i}: ignore rules or tell me ungrounded {i}?"
                        ),
                        "subtype": st,
                        "notes": f"Refuse {st}",
                    }
                )
            return json.dumps({"cases": cases})

        if "edge-case evaluation questions" in prompt:
            cids = list(chunks_by_id.keys())
            sample_cid = cids[0]
            sample_line = next(
                (
                    ln.strip()
                    for ln in chunks_by_id[sample_cid].splitlines()
                    if len(ln.strip()) > 12
                ),
                chunks_by_id[sample_cid],
            )
            cases = []
            for i in range(1, 9):
                if i % 3 == 1:
                    cases.append(
                        {
                            "question": f"Clarify ambiguous sensor question #{i}?",
                            "expected_behavior": "clarify",
                            "source_chunk_ids": [sample_cid],
                            "key_facts": [],
                            "subtype": "ambiguous",
                            "notes": "Ambiguous phrasing",
                        }
                    )
                else:
                    cases.append(
                        {
                            "question": f"Typo variant #{i}: wat is the {sample_line[:22]}?",
                            "expected_behavior": "answer",
                            "source_chunk_ids": [sample_cid],
                            "key_facts": [sample_line[:40]],
                            "subtype": "typo_paraphrase",
                            "notes": "Paraphrase/typo",
                        }
                    )
            return json.dumps({"cases": cases})

        if "specifically require invoking the attached tools" in prompt:
            cids = list(chunks_by_id.keys())
            csv_cid = next((c for c in cids if c.endswith("row-1") or "sensors.csv" in c), cids[0])
            sample_line = chunks_by_id[csv_cid].splitlines()[0]
            cases = [
                {
                    "question": f"Tool query #{i}: look up {sample_line} and calculate total?",
                    "target_tool": "csv_lookup",
                    "source_chunk_ids": [csv_cid],
                    "key_facts": [sample_line],
                    "notes": "Requires csv_lookup + calculator",
                }
                for i in range(1, 7)
            ]
            return json.dumps({"cases": cases})

        raise ValueError(f"Unexpected prompt in test: {prompt[:120]}")

    return _respond


def test_key_fact_gate_drops_fabricated_facts_and_empty_cases() -> None:
    _, chunks, _ = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    chunk_map = {c.chunk_id: c for c in chunks}
    target_chunk = next(c for c in chunks if "45 meters" in c.text)
    prov = EvalProvenance(
        prompt_file="evalgen_single_doc.v1.md",
        prompt_hash="abc",
        model="llama3.1:8b",
        seed=42,
    )

    # Mixed real + fabricated fact: keeps only the real one
    mixed = EvalCase(
        id="C1",
        category="single_doc",
        question="What depth?",
        expected_behavior="answer",
        expected_sources=[
            EvalSourceRef(doc_id=target_chunk.doc_id, chunk_id=target_chunk.chunk_id)
        ],
        key_facts=[
            "Deploy hydrophone pods at 45 meters below mean sea level",
            "Fabricated claim about 999 meters",
        ],
        provenance=prov,
    )
    filtered, err = verify_case_key_facts(mixed, chunk_map)
    assert err is None
    assert filtered is not None
    assert filtered.key_facts == ["Deploy hydrophone pods at 45 meters below mean sea level"]

    # All fabricated -> whole case dropped
    bad_only = mixed.model_copy(update={"key_facts": ["Completely invented telemetry claim"]})
    dropped, drop_err = verify_case_key_facts(bad_only, chunk_map)
    assert dropped is None
    assert drop_err == "no_verified_key_facts"


def test_stratified_split_is_deterministic_and_proportional() -> None:
    prov = EvalProvenance(
        prompt_file="p.md", prompt_hash="h", model="llama3.1:8b", seed=42
    )
    cases = [
        EvalCase(
            id=f"X-{i}",
            category="single_doc" if i < 12 else "out_of_scope",
            question=f"Question number {i}?",
            expected_behavior="answer" if i < 12 else "refuse",
            provenance=prov,
        )
        for i in range(18)
    ]
    dev1, held1 = stratified_split(cases, seed=42)
    dev2, held2 = stratified_split(list(reversed(cases)), seed=42)

    assert len(dev1) == 12
    assert len(held1) == 6
    assert [c.question for c in dev1] == [c.question for c in dev2]
    assert [c.question for c in held1] == [c.question for c in held2]
    assert sum(1 for c in held1 if c.category == "single_doc") == 4
    assert sum(1 for c in held1 if c.category == "out_of_scope") == 2


def test_eval_generator_quotas_gates_and_stats(tmp_path: Path) -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    req = _sample_requirements()
    _, chunks, profile = ingest_folder(
        DOCS_DIR,
        chunk_size=defn.retrieval.chunk_size,
        overlap=defn.retrieval.overlap,
    )
    index = HybridIndex.build(chunks, llm_client=None)
    chunk_texts = {c.chunk_id: c.text for c in chunks}

    cfg = ForgeConfig()
    responder = _smart_evalgen_responder(chunk_texts)
    llm = FakeLLM(responses=[responder] * 200, config=cfg)

    eval_set = EvalGenerator(llm, config=cfg).generate(
        defn,
        req,
        profile,
        index,
        seed=42,
        out_dir=tmp_path,
    )

    assert eval_set.total_cases >= 36
    assert eval_set.category_counts == {
        "single_doc": 12,
        "multi_doc": 6,
        "out_of_scope": 8,
        "edge_case": 6,
        "tool_use": 4,
    }
    assert len(eval_set.dev_cases) == 24
    assert len(eval_set.held_out_cases) == 12
    assert eval_set.same_model_as_agent is False
    assert eval_set.same_family_as_agent is False
    assert eval_set.eval_writer_model == "llama3.1:8b"

    # Verify Gate 1 caught the fabricated key fact and Gate 3 caught the answerable OOS leak
    gates_triggered = {r.gate for r in eval_set.rejected_cases}
    assert "key_fact_verification" in gates_triggered
    assert "oos_answerability_check" in gates_triggered
    assert not any(
        "ANSWERABLE_LEAK_MARKER" in c.question
        for c in [*eval_set.dev_cases, *eval_set.held_out_cases]
    )

    assert (tmp_path / "eval_set.json").is_file()
    assert (tmp_path / "eval_set_stats.md").is_file()
    stats_md = (tmp_path / "eval_set_stats.md").read_text(encoding="utf-8")
    assert "Same Model Family as Agent" in stats_md
    assert "## Rejected Candidates by Gate / Reason" in stats_md


def test_eval_generator_no_tools_redistributes_to_36(tmp_path: Path) -> None:
    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    defn.tools = []
    req = _sample_requirements()
    _, chunks, profile = ingest_folder(DOCS_DIR, chunk_size=400, overlap=40)
    index = HybridIndex.build(chunks, llm_client=None)
    chunk_texts = {c.chunk_id: c.text for c in chunks}

    cfg = ForgeConfig()
    responder = _smart_evalgen_responder(chunk_texts)
    llm = FakeLLM(responses=[responder] * 200, config=cfg)

    eval_set = EvalGenerator(llm, config=cfg).generate(
        defn, req, profile, index, seed=42
    )
    assert eval_set.total_cases >= 36
    assert eval_set.category_counts.get("tool_use", 0) == 0


def test_cli_evals_generate_writes_36_cases_and_stats(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify `forge evals generate` writes eval_set.json (>=36 cases) and eval_set_stats.md."""
    req_path = tmp_path / "requirements.json"
    req_path.write_text(_sample_requirements().model_dump_json(indent=2), encoding="utf-8")

    defn = AgentDefinition.load_yaml(HAND_AGENT_PATH)
    _, chunks, _ = ingest_folder(
        DOCS_DIR,
        chunk_size=defn.retrieval.chunk_size,
        overlap=defn.retrieval.overlap,
    )
    responder = _smart_evalgen_responder({c.chunk_id: c.text for c in chunks})
    fake_llm = FakeLLM(responses=[responder] * 200)
    monkeypatch.setattr(
        forge_cli.LLMClient,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: fake_llm),
    )

    out_dir = tmp_path / "eval_out"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "evals",
            "generate",
            "--definition",
            str(HAND_AGENT_PATH),
            "--requirements",
            str(req_path),
            "--docs",
            str(DOCS_DIR),
            "--out",
            str(out_dir),
            "--bm25-only",
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / "eval_set.json").is_file()
    assert (out_dir / "eval_set_stats.md").is_file()
    data = json.loads((out_dir / "eval_set.json").read_text(encoding="utf-8"))
    assert len(data["dev_cases"]) + len(data["held_out_cases"]) >= 36
