"""Smoke and CLI integration tests for ForgeAI."""

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from forge.cli import app

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "phase1_sample"


def test_cli_help_via_runner() -> None:
    """Verify `forge --help` renders help text via Typer's CliRunner."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ForgeAI" in result.stdout
    assert "ingest" in result.stdout
    assert "search" in result.stdout


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
