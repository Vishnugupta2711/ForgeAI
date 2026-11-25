"""Traceable quality report generation and numeric narrative lint."""

from forge.report.builder import build_report
from forge.report.lint import collect_allowed_numbers, lint_narrative_numbers

__all__ = [
    "build_report",
    "collect_allowed_numbers",
    "lint_narrative_numbers",
]
