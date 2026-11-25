"""Data models and markdown renderer for ForgeAI intake requirements."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from forge.groundability.models import GroundabilityReport, GroundabilityStatus


class IntakeQuestion(BaseModel):
    """A single clarifying question proposed during intake."""

    id: str
    question: str = Field(min_length=1)
    why: str = ""


class IntakeAnswer(BaseModel):
    """Collected answer for an intake clarifying question."""

    question_id: str
    question: str
    answer: str
    source: Literal["interactive", "answers_file", "auto"]


class RequirementTopic(BaseModel):
    """A structured must-answer topic with a traceable REQ identifier."""

    id: str
    title: str
    description: str
    priority: Literal["high", "medium", "low"] = "high"
    groundability_status: GroundabilityStatus | None = None


class RefusalTopic(BaseModel):
    """A structured out-of-scope or ungrounded topic that the agent must refuse."""

    id: str
    title: str
    reason: str
    from_gap: bool = False


class CapabilityNeed(BaseModel):
    """A required tool or capability linked back to REQ identifiers."""

    name: str
    reason: str
    linked_req_ids: list[str] = Field(default_factory=list)
    capability_tag: str | None = None


class RequirementsDocument(BaseModel):
    """Complete structured requirements document produced by intake + groundability."""

    brief: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[RequirementTopic] = Field(default_factory=list)
    must_refuse_topics: list[RefusalTopic] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    capabilities_needed: list[CapabilityNeed] = Field(default_factory=list)
    clarifying_qa: list[IntakeAnswer] = Field(default_factory=list)
    groundability_report: GroundabilityReport | None = None

    def to_markdown(self) -> str:
        """Render requirements and groundability findings to human-readable Markdown."""
        lines: list[str] = ["# ForgeAI Requirements Specification", ""]

        if self.groundability_report and self.groundability_report.has_gaps:
            lines.extend(
                [
                    "## ⚠️ Groundability Gaps Flagged",
                    "",
                    (
                        "One or more requested topics lacked full document support and were "
                        "either moved to **Must Refuse** or narrowed in scope:"
                    ),
                    "",
                ]
            )
            for gap in self.groundability_report.flagged_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        lines.extend(
            [
                "## Brief",
                self.brief.strip(),
                "",
                "## Scope, Audience & Tone",
                f"- **Scope**: {self.scope.strip()}",
                f"- **Audience**: {self.audience.strip()}",
                f"- **Tone**: {self.tone.strip()}",
                "",
                "## Must-Answer Topics",
            ]
        )
        if self.must_answer_topics:
            for topic in self.must_answer_topics:
                status_badge = (
                    f" [{topic.groundability_status}]" if topic.groundability_status else ""
                )
                lines.append(
                    f"- **{topic.id}** ({topic.priority}){status_badge}: "
                    f"**{topic.title}** — {topic.description}"
                )
        else:
            lines.append("- (none)")

        lines.extend(["", "## Must-Refuse Topics"])
        if self.must_refuse_topics:
            for ref in self.must_refuse_topics:
                gap_tag = " [GROUNDABILITY GAP]" if ref.from_gap else ""
                lines.append(f"- **{ref.id}**{gap_tag}: **{ref.title}** — {ref.reason}")
        else:
            lines.append("- (none)")

        lines.extend(["", "## Capabilities Needed"])
        if self.capabilities_needed:
            for cap in self.capabilities_needed:
                linked = ", ".join(cap.linked_req_ids) if cap.linked_req_ids else "none"
                lines.append(f"- **{cap.name}** (linked: {linked}): {cap.reason}")
        else:
            lines.append("- (none)")

        lines.extend(["", "## Assumptions"])
        if self.assumptions:
            for item in self.assumptions:
                lines.append(f"- {item}")
        else:
            lines.append("- (none)")

        lines.extend(["", "## Clarifying Q&A"])
        if self.clarifying_qa:
            for qa in self.clarifying_qa:
                lines.append(
                    f"- **{qa.question_id}** ({qa.source}): {qa.question} → {qa.answer}"
                )
        else:
            lines.append("- (none)")

        lines.extend(["", "## Groundability Report"])
        if self.groundability_report is None:
            lines.append("- Groundability check not run.")
        else:
            rep = self.groundability_report
            lines.append(
                f"- Summary: `{rep.supported_count}` SUPPORTED, "
                f"`{rep.partial_count}` PARTIAL, `{rep.unsupported_count}` UNSUPPORTED "
                f"(strict={rep.strict_mode})"
            )
            for item in rep.topics:
                verified_count = sum(1 for q in item.quotes if q.verified_substring)
                lines.append(
                    f"- **{item.topic_id} — {item.title}**: **{item.status}** "
                    f"(coverage={item.coverage_score:.2f}, overlap={item.key_term_overlap:.2f}, "
                    f"verified_quotes={verified_count}) — {item.reasoning}"
                )
                for q in item.quotes:
                    mark = "verified" if q.verified_substring else "UNVERIFIED"
                    lines.append(f"  - quote [{mark}] ({q.chunk_id}): \"{q.quote}\"")
                if item.gap_note:
                    lines.append(f"  - gap_note: {item.gap_note}")

        lines.append("")
        return "\n".join(lines)

    def save(self, out_dir: Path | str) -> tuple[Path, Path]:
        """Write requirements.json and requirements.md into out_dir."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "requirements.json"
        md_path = target / "requirements.md"
        json_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path
