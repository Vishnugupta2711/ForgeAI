"""Numeric and case-ID lint guards for generated report narratives."""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"(?<![a-zA-Z0-9_-])(\d+(?:,\d{3})*(?:\.\d+)?)(?![a-zA-Z0-9_-])")
_CASE_ID_RE = re.compile(r"\b(?:DEV|HELD|EVAL|REQ|REF|TOP|CAN)-[A-Z0-9-]+\b")


def _canon_num(raw: str) -> str:
    cleaned = raw.replace(",", "")
    try:
        val = float(cleaned)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def collect_allowed_numbers(obj: Any) -> set[str]:
    """Recursively collect all canonical numeric values present in result dicts/lists."""
    allowed: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            allowed.add(_canon_num(str(node)))
            # Also allow percentage form (e.g. 0.75 -> 75, 75.0) for 0..1 rates
            if 0.0 <= float(node) <= 1.0:
                pct = float(node) * 100.0
                allowed.add(_canon_num(f"{pct:.1f}"))
                allowed.add(_canon_num(f"{pct:.2f}"))
                allowed.add(_canon_num(f"{round(pct)}"))
            return
        if isinstance(node, str):
            for match in _NUM_RE.findall(node):
                allowed.add(_canon_num(match))
            return
        if isinstance(node, dict):
            for val in node.values():
                _walk(val)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(obj)
    return allowed


def lint_narrative_numbers(
    narrative_text: str,
    allowed_numbers: set[str],
    *,
    valid_case_ids: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Verify every number in narrative_text exists in allowed_numbers and check case IDs.

    Returns (passed, violations).
    """
    violations: list[str] = []

    # Ignore digits inside explicit case IDs like DEV-001 (negative lookbehind handles '-')
    for match in _NUM_RE.findall(narrative_text):
        canon = _canon_num(match)
        if canon not in allowed_numbers:
            violations.append(f"Invented or unverified number '{match}' in narrative.")

    if valid_case_ids:
        cited = set(_CASE_ID_RE.findall(narrative_text))
        if not cited:
            violations.append("Narrative failed to cite any valid case IDs.")
        unknown = cited - valid_case_ids
        if unknown:
            violations.append(f"Narrative cited unknown case IDs: {sorted(unknown)}")

    return len(violations) == 0, violations
