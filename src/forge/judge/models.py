"""Data models for ForgeAI Judge verdicts, scorecards, and canary reliability."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from forge.agent.schema import AnswerRecord
from forge.evals.models import EvalCategory, ExpectedBehavior


class ClaimVerification(BaseModel):
    """Groundedness verification for a single extracted claim."""

    claim: str
    supported: bool
    quote: str = ""
    verified_substring: bool = False
    status: str = "supported"  # "supported" | "unsupported" | "uncertain"
    method: str = "deterministic"  # "deterministic" | "llm" | "fallback"
    reason: str = ""


class CaseVerdict(BaseModel):
    """Judge verdict for a single evaluation case."""

    case_id: str
    category: EvalCategory
    expected_behavior: ExpectedBehavior
    grounded: bool
    correct: bool
    refusal_ok: bool
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    uncertain: bool = False
    uncertain_claims: int = 0
    used_fallback: bool = False
    passed_via_fallback: bool = False
    key_fact_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    citations_valid: bool = True
    ungrounded_numbers: list[str] = Field(default_factory=list)
    claim_results: list[ClaimVerification] = Field(default_factory=list)
    position_bias_disagreement: bool = False
    answer_length_chars: int = 0


class EvalCaseResult(BaseModel):
    """Combined execution + verdict record saved in results.json."""

    case_id: str
    category: EvalCategory
    split: str
    question: str
    expected_behavior: ExpectedBehavior
    answer_record: AnswerRecord
    verdict: CaseVerdict


class CategoryMetric(BaseModel):
    """Pass rate and Wilson 95% confidence interval for one eval category."""

    category: str
    total: int
    passed: int
    pass_rate: float
    wilson_low: float
    wilson_high: float


class EvalScorecard(BaseModel):
    """Aggregate scorecard and case results for an evaluation run."""

    split: str
    agent_name: str
    agent_model: str
    judge_model: str
    same_model_as_agent: bool
    same_family_as_agent: bool
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_cases: int
    passed_cases: int
    overall_pass_rate: float
    overall_wilson_low: float
    overall_wilson_high: float
    small_sample_note: str = (
        "Evaluation categories contain small finite sample sizes; Wilson 95% confidence "
        "intervals quantify uncertainty and should be read alongside case-level failures."
    )
    suspicion_flag: bool = False
    suspicion_note: str | None = None
    length_pass_correlation: float | None = None
    category_metrics: list[CategoryMetric] = Field(default_factory=list)
    results: list[EvalCaseResult] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render a concise, honest scorecard in Markdown."""
        lines = [
            "# Evaluation Scorecard",
            "",
            f"- **Split**: `{self.split}`",
            f"- **Agent**: `{self.agent_name}` (`{self.agent_model}`)",
            f"- **Judge Model**: `{self.judge_model}`",
            f"- **Same Model / Family as Agent**: "
            f"`{self.same_model_as_agent}` / `{self.same_family_as_agent}`",
            (
                f"- **Overall Pass Rate**: `{self.passed_cases}/{self.total_cases}` "
                f"(`{self.overall_pass_rate * 100:.1f}%`, "
                f"Wilson 95% CI: [{self.overall_wilson_low * 100:.1f}%, "
                f"{self.overall_wilson_high * 100:.1f}%])"
            ),
            (
                f"- **Answer Length vs Pass Correlation (Pearson r)**: "
                f"`{self.length_pass_correlation:.3f}`"
                if self.length_pass_correlation is not None
                else "- **Answer Length vs Pass Correlation (Pearson r)**: `n/a`"
            ),
            "",
            f"> Note: {self.small_sample_note}",
            "",
        ]
        if self.same_model_as_agent or self.same_family_as_agent:
            lines.append(
                "> **WARNING (Self-Preference Risk)**: Judge and agent share the same model "
                "or model family; scores may reflect self-preference bias."
            )
            lines.append("")
        if self.suspicion_flag and self.suspicion_note:
            lines.append(f"> **SUSPICION FLAG**: {self.suspicion_note}")
            lines.append("")

        lines.extend(
            [
                "## Per-Category Pass Rates (Wilson 95% CI)",
                "| Category | Passed | Total | Pass Rate | Wilson 95% CI |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for m in self.category_metrics:
            lines.append(
                f"| {m.category} | {m.passed} | {m.total} | "
                f"{m.pass_rate * 100:.1f}% | "
                f"[{m.wilson_low * 100:.1f}%, {m.wilson_high * 100:.1f}%] |"
            )

        failures = [r for r in self.results if not r.verdict.passed]
        lines.extend(["", f"## Failed Cases ({len(failures)})"])
        if failures:
            for item in failures:
                v = item.verdict
                lines.append(
                    f"- **{v.case_id}** (`{v.category}`): "
                    f"grounded={v.grounded}, correct={v.correct}, refusal_ok={v.refusal_ok}, "
                    f"uncertain={v.uncertain} — {'; '.join(v.reasons)}"
                )
        else:
            lines.append("- None")

        lines.append("")
        return "\n".join(lines)

    def save(self, out_dir: Path | str) -> tuple[Path, Path]:
        """Persist results.json and scorecard.md into out_dir."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        results_path = target / "results.json"
        scorecard_path = target / "scorecard.md"
        results_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        scorecard_path.write_text(self.to_markdown(), encoding="utf-8")
        return results_path, scorecard_path


class CanaryMetric(BaseModel):
    """Reliability rates for a single canary category."""

    canary_type: str
    expected_pass: bool
    total: int
    actual_pass: int
    passed_cleanly: int = 0
    passed_via_fallback: int = 0
    false_passes: int
    false_fails: int
    false_pass_rate: float
    false_fail_rate: float


class CanaryCaseResult(BaseModel):
    """Individual canary case evaluation outcome."""

    canary_id: str
    canary_type: str
    expected_pass: bool
    passed_via_fallback: bool = False
    verdict: CaseVerdict


class JudgeReliabilityReport(BaseModel):
    """Report written to judge_reliability.json by `forge judge-selftest`."""

    judge_model: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_canaries: int
    overall_false_pass_rate: float
    overall_false_fail_rate: float
    overall_clean_pass_rate: float = 0.0
    passed_cleanly_count: int = 0
    passed_via_fallback_count: int = 0
    per_canary_type: list[CanaryMetric] = Field(default_factory=list)
    cases: list[CanaryCaseResult] = Field(default_factory=list)

    def save(self, path: Path | str) -> Path:
        """Write judge_reliability.json to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target
