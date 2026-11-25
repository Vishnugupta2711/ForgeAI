"""Data models for ForgeAI failure classification, typed patches, and iteration logs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

FailureCategory = Literal[
    "retrieval_miss",
    "hallucination",
    "wrong_refusal",
    "prompt_defect",
]

PromptSectionName = Literal[
    "role",
    "scope",
    "audience_tone",
    "grounding_rules",
    "refusal_policy",
    "format_rules",
    "extra_rules",
]

PromptPatchOp = Literal["add_rule", "replace_rule", "add_example"]


class ClassifiedFailure(BaseModel):
    """Single primary failure classification for a failed EvalCaseResult."""

    case_id: str
    eval_category: str
    failure_category: FailureCategory
    subtype: str = ""
    evidence: str
    question_preview: str = ""
    reasons: list[str] = Field(default_factory=list)
    used_llm_tiebreaker: bool = False


class RetrievalPatch(BaseModel):
    """Typed patch modifying only specified retrieval settings."""

    kind: Literal["retrieval"] = "retrieval"
    target_fields: dict[str, Any] = Field(default_factory=dict)
    addressed_case_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def triggers_reindex(self) -> bool:
        """True if chunk_size or overlap changed and requires re-chunking."""
        return "chunk_size" in self.target_fields or "overlap" in self.target_fields


class PromptPatch(BaseModel):
    """Typed patch modifying only a single named SystemPromptSections field."""

    kind: Literal["prompt"] = "prompt"
    section: PromptSectionName
    op: PromptPatchOp = "add_rule"
    text: str = Field(min_length=1)
    target_rule_index: int | None = None
    addressed_case_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


OptimizerPatch = Annotated[RetrievalPatch | PromptPatch, Field(discriminator="kind")]


class IterationRecord(BaseModel):
    """Audit record for one critic-optimizer iteration."""

    iteration: int
    score_before: float
    score_after: float
    category_scores_before: dict[str, float] = Field(default_factory=dict)
    category_scores_after: dict[str, float] = Field(default_factory=dict)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    proposed_patches: list[OptimizerPatch] = Field(default_factory=list)
    rejected_by_leak_guard: list[str] = Field(default_factory=list)
    accepted: bool = False
    rolled_back: bool = False
    decision_reason: str = ""


class OptimizationScorecard(BaseModel):
    """Top-level optimizer run summary written to scorecard.json."""

    agent_name: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    baseline_dev_score: float
    final_dev_score: float
    baseline_held_out_score: float
    final_held_out_score: float
    baseline_held_out_by_category: dict[str, float] = Field(default_factory=dict)
    final_held_out_by_category: dict[str, float] = Field(default_factory=dict)
    iterations_completed: int
    stop_reason: str
    iterations: list[IterationRecord] = Field(default_factory=list)

    def save(self, path: Path | str) -> Path:
        """Write scorecard.json to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target
