"""Deterministic rule-gated tool selection from the ForgeAI registry."""

from __future__ import annotations

import re

from pydantic import BaseModel

from forge.agent.schema import AttachedToolSpec
from forge.ingest.models import CorpusProfile
from forge.intake.models import CapabilityNeed, RequirementsDocument
from forge.tools import ToolRegistry

_TOOL_CAPABILITY_ALIASES: dict[str, set[str]] = {
    "csv_lookup": {
        "tabular_lookup",
        "csv_lookup",
        "csv",
        "tabular",
        "table_lookup",
        "structured_data",
        "spreadsheet",
    },
    "calculator": {
        "arithmetic",
        "calculator",
        "math",
        "calculation",
        "calculations",
        "numeric",
        "totals",
    },
    "http_api": {
        "live_external_data",
        "http_api",
        "weather",
        "forecast",
        "external_api",
        "live_weather",
        "geocoding",
    },
    "current_date": {
        "current_date",
        "date",
        "today",
        "calendar",
        "current_time",
    },
}

_LIVE_EXTERNAL_RE = re.compile(
    r"\b(weather|forecast|temperature|precipitation|wind|open-meteo|"
    r"live external|external api|live api|live data|real[- ]time weather)\b",
    re.IGNORECASE,
)

_CALC_INTENT_RE = re.compile(
    r"\b(calculat|arithmetic|sum|total|average|mean|ratio|percent|"
    r"cost|price|budget|margin|formula|math|multiply|divide|difference)\b",
    re.IGNORECASE,
)

_DATE_INTENT_RE = re.compile(
    r"\b(today|current date|today's date|what day|deadline|expir|"
    r"days remaining|as of today|calendar date)\b",
    re.IGNORECASE,
)


class ToolDecision(BaseModel):
    """Audit record for a single candidate tool evaluated by the deterministic gate."""

    tool_name: str
    attached: bool
    proposed_by_llm: bool = False
    justification_requirement_id: str | None = None
    matched_capabilities: list[str] = []
    reason: str


def _matches_tool_capability(tool_name: str, cap: CapabilityNeed) -> bool:
    aliases = _TOOL_CAPABILITY_ALIASES.get(tool_name, {tool_name})
    normalized_name = cap.name.strip().lower().replace("-", "_").replace(" ", "_")
    normalized_tag = (cap.capability_tag or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized_name in aliases or (bool(normalized_tag) and normalized_tag in aliases)


def gate_tools(
    requirements: RequirementsDocument,
    profile: CorpusProfile,
    registry: ToolRegistry,
    *,
    llm_proposed: list[str] | None = None,
) -> tuple[list[AttachedToolSpec], list[ToolDecision]]:
    """Select only tools justified by active REQ IDs and supported by docs/brief.

    An LLM may propose tools, but deterministic code gates make the final decision.
    """
    proposed_set = {p.strip() for p in (llm_proposed or []) if p.strip()}
    active_req_map = {req.id: req for req in requirements.must_answer_topics}
    has_tabular_doc = any(
        doc.doc_type in {"csv", "xlsx"} or doc.csv_profile is not None
        for doc in profile.documents
    )
    has_any_docs = profile.loaded_docs > 0

    attached_specs: list[AttachedToolSpec] = []
    decisions: list[ToolDecision] = []

    for spec in registry.list_specs():
        t_name = spec.name
        is_proposed = t_name in proposed_set

        matching_caps: list[CapabilityNeed] = []
        valid_req_ids: list[str] = []
        for cap in requirements.capabilities_needed:
            if _matches_tool_capability(t_name, cap):
                linked = [rid for rid in cap.linked_req_ids if rid in active_req_map]
                if linked:
                    matching_caps.append(cap)
                    for rid in linked:
                        if rid not in valid_req_ids:
                            valid_req_ids.append(rid)

        if not matching_caps or not valid_req_ids:
            reason = (
                "Rejected: not requested in requirements.capabilities_needed "
                "with an active linked REQ ID."
            )
            if is_proposed:
                reason = (
                    "Rejected LLM proposal: requirements.capabilities_needed has no "
                    "matching capability linked to an active must-answer REQ ID."
                )
            decisions.append(
                ToolDecision(
                    tool_name=t_name,
                    attached=False,
                    proposed_by_llm=is_proposed,
                    reason=reason,
                )
            )
            continue

        linked_text = " ".join(
            f"{active_req_map[rid].title} {active_req_map[rid].description}"
            for rid in valid_req_ids
        )
        cap_text = " ".join(f"{c.name} {c.reason}" for c in matching_caps)
        combined_intent_text = f"{requirements.brief} {linked_text} {cap_text}"

        # Per-tool environment/corpus/brief gates
        gate_error: str | None = None
        if t_name == "csv_lookup":
            if not has_tabular_doc:
                gate_error = "Rejected: corpus contains no tabular (CSV/XLSX) document."
        elif t_name == "http_api":
            brief_and_reqs = f"{requirements.brief} {linked_text}"
            if not _LIVE_EXTERNAL_RE.search(brief_and_reqs):
                gate_error = (
                    "Rejected: brief and active requirements do not explicitly "
                    "require live external data/weather API."
                )
        elif t_name == "calculator":
            if not has_any_docs:
                gate_error = "Rejected: corpus has no loaded documents."
            elif not (has_tabular_doc or _CALC_INTENT_RE.search(combined_intent_text)):
                gate_error = (
                    "Rejected: neither tabular documents nor numeric calculation intent "
                    "is present in brief/requirements."
                )
        elif t_name == "current_date":
            if not _DATE_INTENT_RE.search(combined_intent_text):
                gate_error = (
                    "Rejected: brief and active requirements do not indicate a "
                    "current-date or time-relative need."
                )

        if gate_error is not None:
            decisions.append(
                ToolDecision(
                    tool_name=t_name,
                    attached=False,
                    proposed_by_llm=is_proposed,
                    matched_capabilities=[c.name for c in matching_caps],
                    reason=gate_error,
                )
            )
            continue

        justification = ", ".join(valid_req_ids)
        attached_specs.append(
            AttachedToolSpec(
                name=t_name,
                config={},
                justification_requirement_id=justification,
            )
        )
        decisions.append(
            ToolDecision(
                tool_name=t_name,
                attached=True,
                proposed_by_llm=is_proposed,
                justification_requirement_id=justification,
                matched_capabilities=[c.name for c in matching_caps],
                reason=(
                    f"Attached: capability linked to {justification} "
                    "and supported by corpus/brief."
                ),
            )
        )

    return attached_specs, decisions
