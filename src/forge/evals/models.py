"""Data models for source-verified evaluation datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EvalCategory = Literal["single_doc", "multi_doc", "out_of_scope", "edge_case", "tool_use"]
ExpectedBehavior = Literal["answer", "refuse", "clarify"]
EvalSplit = Literal["dev", "held_out"]


class EvalSourceRef(BaseModel):
    """Gold provenance reference pointing to a corpus document and chunk."""

    doc_id: str
    chunk_id: str


class EvalProvenance(BaseModel):
    """Generation provenance attached to each EvalCase."""

    prompt_file: str
    prompt_hash: str
    model: str
    seed: int


class EvalCase(BaseModel):
    """Single verified evaluation case."""

    id: str
    category: EvalCategory
    split: EvalSplit = "dev"
    question: str = Field(min_length=1)
    expected_behavior: ExpectedBehavior
    expected_sources: list[EvalSourceRef] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    notes: str = ""
    provenance: EvalProvenance


class RejectedCaseLog(BaseModel):
    """Audit log entry for a candidate eval case rejected by a quality gate."""

    question: str
    category: EvalCategory
    gate: str
    reason: str


class EvalSet(BaseModel):
    """Complete evaluation dataset with dev/held_out split and rejection audit trail."""

    agent_name: str
    eval_writer_model: str
    agent_model: str
    same_model_as_agent: bool
    same_family_as_agent: bool
    seed: int
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    dev_cases: list[EvalCase] = Field(default_factory=list)
    held_out_cases: list[EvalCase] = Field(default_factory=list)
    rejected_cases: list[RejectedCaseLog] = Field(default_factory=list)
    category_counts: dict[str, int] = Field(default_factory=dict)
    dev_category_counts: dict[str, int] = Field(default_factory=dict)
    held_out_category_counts: dict[str, int] = Field(default_factory=dict)
    rejected_counts_by_reason: dict[str, int] = Field(default_factory=dict)
    scaled_quotas_note: str | None = None

    @property
    def total_cases(self) -> int:
        """Total number of accepted cases across dev and held_out."""
        return len(self.dev_cases) + len(self.held_out_cases)

    def to_stats_markdown(self) -> str:
        """Render a concise Markdown summary of dataset counts and rejections."""
        lines = [
            "# Evaluation Set Statistics",
            "",
            f"- **Agent**: `{self.agent_name}` (`{self.agent_model}`)",
            f"- **Eval Writer Model**: `{self.eval_writer_model}`",
            f"- **Same Model as Agent**: `{self.same_model_as_agent}`",
            f"- **Same Model Family as Agent**: `{self.same_family_as_agent}`",
            f"- **Seed**: `{self.seed}`",
            (
                f"- **Total Verified Cases**: `{self.total_cases}` "
                f"(dev=`{len(self.dev_cases)}`, held_out=`{len(self.held_out_cases)}`)"
            ),
            f"- **Total Rejected Candidates**: `{len(self.rejected_cases)}`",
        ]
        if self.scaled_quotas_note:
            lines.append(f"- **Quota Scaling Note**: {self.scaled_quotas_note}")
        lines.extend(
            [
                "",
                "## Category Counts",
                "| Category | Total | Dev | Held-Out |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        all_cats = sorted(
            set(self.category_counts)
            | set(self.dev_category_counts)
            | set(self.held_out_category_counts)
        )
        for cat in all_cats:
            lines.append(
                f"| {cat} | {self.category_counts.get(cat, 0)} | "
                f"{self.dev_category_counts.get(cat, 0)} | "
                f"{self.held_out_category_counts.get(cat, 0)} |"
            )

        lines.extend(["", "## Rejected Candidates by Gate / Reason"])
        if self.rejected_counts_by_reason:
            for reason, count in sorted(self.rejected_counts_by_reason.items()):
                lines.append(f"- **{reason}**: {count}")
        else:
            lines.append("- None")

        lines.append("")
        return "\n".join(lines)

    def save(self, out_dir: Path | str) -> tuple[Path, Path]:
        """Persist eval_set.json and eval_set_stats.md into out_dir."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "eval_set.json"
        stats_path = target / "eval_set_stats.md"
        json_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        stats_path.write_text(self.to_stats_markdown(), encoding="utf-8")
        return json_path, stats_path
