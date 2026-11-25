"""Smoke and CLI integration tests for ForgeAI."""

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from forge import cli as forge_cli
from forge.agent import AgentDefinition, AgentRuntime
from forge.cli import app
from forge.ingest import ingest_folder
from forge.llm import FakeLLM
from forge.retrieval import HybridIndex
from forge.tools import ToolContext

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "phase1_sample"
DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docs"
HAND_AGENT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hand_agent.yaml"


def test_cli_help_via_runner() -> None:
    """Verify `forge --help` renders help text via Typer's CliRunner."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ForgeAI" in result.stdout
    assert "ingest" in result.stdout
    assert "search" in result.stdout
    assert "chat" in result.stdout
    assert "intake" in result.stdout


def test_cli_help_via_entrypoint() -> None:
    """Verify the installed `forge` console script runs with `--help`."""
    forge_bin = Path(sys.executable).parent / "forge"
    cmd = [str(forge_bin), "--help"] if forge_bin.exists() else ["forge", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "ForgeAI" in proc.stdout


def test_cli_ingest_and_search(tmp_path: Path) -> None:
    """Verify `forge ingest` and `forge search --bm25-only` work end-to-end."""
    runner = CliRunner()
    run_dir = tmp_path / "run"

    ingest_res = runner.invoke(
        app,
        ["ingest", str(FIXTURE_DIR), "--run-dir", str(run_dir)],
    )
    assert ingest_res.exit_code == 0
    assert "acoustic_guide.md" in ingest_res.stdout
    assert "sensors.csv" in ingest_res.stdout
    assert (run_dir / "profile.json").is_file()

    search_res = runner.invoke(
        app,
        [
            "search",
            str(FIXTURE_DIR),
            "hydrophone deployment depth 45 meters",
            "--bm25-only",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert search_res.exit_code == 0
    assert "acoustic_guide.md" in search_res.stdout
    assert "score=" in search_res.stdout
    assert (run_dir / "index" / "meta.json").is_file()


def test_cli_chat_multiturn_with_trace_and_refusal(monkeypatch) -> None:
    """Verify `forge chat` loads hand_agent.yaml, prints citations, trace, and refuses OOS."""
    fake_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "action": "final",
                    "answer": "Deploy hydrophone pods at 45 meters below mean sea level.",
                    "citations": [
                        {
                            "doc_id": "acoustic_guide.md",
                            "chunk_id": "acoustic_guide.md::c1",
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": 0.94,
                }
            ),
            json.dumps(
                {
                    "action": "final",
                    "answer": "I cannot answer salary questions; they are not in the docs.",
                    "citations": [],
                    "refused": True,
                    "refusal_reason": "Salary information is outside the grounded scope.",
                    "confidence": 0.98,
                }
            ),
        ]
    )

    def _fake_build_runtime(
        definition: AgentDefinition,
        docs: Path,
        *,
        run_dir: Path | None,
        bm25_only: bool,
        llm_client=None,
    ) -> AgentRuntime:
        ingest_res, chunks, _ = ingest_folder(
            docs,
            chunk_size=definition.retrieval.chunk_size,
            overlap=definition.retrieval.overlap,
        )
        index = HybridIndex.build(chunks, llm_client=fake_llm)
        tool_ctx = ToolContext.from_documents(ingest_res.documents)
        return AgentRuntime(definition, index, llm_client=fake_llm, tool_context=tool_ctx)

    monkeypatch.setattr(forge_cli, "_build_runtime_for_chat", _fake_build_runtime)

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["chat", str(HAND_AGENT_PATH), "--docs", str(DOCS_DIR), "--trace"],
        input="What depth should hydrophone pods be deployed?\nWhat is the CEO salary?\nexit\n",
    )
    assert res.exit_code == 0
    assert "marine-sensor-assistant" in res.stdout
    assert "ANSWERED" in res.stdout
    assert "acoustic_guide.md" in res.stdout
    assert "Trace:" in res.stdout
    assert "REFUSED" in res.stdout


def test_cli_intake_auto_writes_requirements_and_strict_exits(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify `forge intake --brief ... --docs ... --auto --out ...` writes artifacts."""
    brief_file = tmp_path / "brief.txt"
    brief_file.write_text(
        "I need an assistant for hydrophone deployment depth and calibration schedules.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "runs" / "tmp"

    fake_llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "questions": [
                        {"id": "Q1", "question": "Preferred answer length?", "why": "tone"}
                    ]
                }
            ),
            json.dumps(
                {"answers": [{"question_id": "Q1", "answer": "Keep answers concise."}]}
            ),
            json.dumps(
                {
                    "scope": "Hydrophone depth and buoy calibration.",
                    "audience": "Field technicians",
                    "tone": "Concise",
                    "must_answer_topics": [
                        {
                            "id": "REQ-001",
                            "title": "Deployment Depth",
                            "description": "Hydrophone pod deployment depth",
                            "priority": "high",
                        }
                    ],
                    "must_refuse_topics": [],
                    "assumptions": [],
                    "capabilities_needed": [],
                }
            ),
            json.dumps(
                {
                    "topics": [
                        {
                            "topic_id": "TOP-001",
                            "title": "Deployment Depth",
                            "description": "Hydrophone pod deployment depth",
                            "source_req_id": "REQ-001",
                            "is_implicit": False,
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "status": "SUPPORTED",
                    "quotes": [
                        {
                            "chunk_id": "acoustic_guide.md::c2",
                            "quote": "Deploy hydrophone pods at 45 meters below mean sea level",
                        }
                    ],
                    "reasoning": "Supported by acoustic_guide.md.",
                    "gap_note": None,
                }
            ),
        ]
    )

    monkeypatch.setattr(
        forge_cli.LLMClient,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: fake_llm),
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "intake",
            "--brief",
            str(brief_file),
            "--docs",
            str(DOCS_DIR),
            "--auto",
            "--bm25-only",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / "requirements.json").is_file()
    assert (out_dir / "requirements.md").is_file()
    md_content = (out_dir / "requirements.md").read_text(encoding="utf-8")
    assert "## Groundability Report" in md_content
    assert "Auto-assumed (Q1:" in md_content
