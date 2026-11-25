"""Smoke tests for the ForgeAI CLI entry point."""

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from forge.cli import app


def test_cli_help_via_runner() -> None:
    """Verify `forge --help` renders help text via Typer's CliRunner."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ForgeAI" in result.stdout
    assert "version" in result.stdout


def test_cli_help_via_entrypoint() -> None:
    """Verify the installed `forge` console script runs with `--help`."""
    forge_bin = Path(sys.executable).parent / "forge"
    cmd = [str(forge_bin), "--help"] if forge_bin.exists() else ["forge", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "ForgeAI" in proc.stdout
