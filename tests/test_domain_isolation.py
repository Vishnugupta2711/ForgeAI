"""Static domain-isolation test ensuring src/ and prompts/ contain zero domain-specific terms."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
PROMPTS_DIR = ROOT / "prompts"

# Explicit allowlist of (relative_path, line_substring) if ever genuinely needed
ALLOWLIST: set[tuple[str, str]] = set()

DOMAIN_TERMS_RE = re.compile(
    r"(?<![a-zA-Z0-9_])("
    r"hr|ship|ships|supplier|suppliers|leave|payroll|salary|salaries|"
    r"maritime|vessel|vessels|spares|shore_staff|hazmat|crew"
    r")(?![a-zA-Z0-9_])",
    re.IGNORECASE,
)


def test_no_domain_terms_in_src_or_prompts() -> None:
    """Fail if any domain-specific term appears inside src/ or prompts/."""
    violations: list[str] = []

    targets = [
        *sorted(SRC_DIR.rglob("*.py")),
        *sorted(PROMPTS_DIR.rglob("*.md")),
    ]
    for path in targets:
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = DOMAIN_TERMS_RE.search(line)
            if not match:
                continue
            if any(rel == a_path and a_sub in line for a_path, a_sub in ALLOWLIST):
                continue
            violations.append(f"{rel}:{lineno}: matched '{match.group(0)}' -> {line.strip()}")

    assert not violations, "Domain-specific terms found in src/ or prompts/:\n" + "\n".join(
        violations
    )
