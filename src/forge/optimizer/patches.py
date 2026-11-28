"""Typed patch operators mutating only targeted AgentDefinition fields."""

from __future__ import annotations

from typing import Any

from forge.agent.schema import AgentDefinition, AgentRetrievalSettings
from forge.optimizer.models import (
    ClassifiedFailure,
    OptimizerPatch,
    PromptPatch,
    RetrievalPatch,
)

ALLOWED_RETRIEVAL_FIELDS = frozenset(
    {
        "top_k",
        "chunk_size",
        "overlap",
        "hybrid_alpha",
        "rerank",
        "query_rewrite",
        "min_score",
    }
)


def apply_patch(definition: AgentDefinition, patch: OptimizerPatch) -> AgentDefinition:
    """Apply a single typed patch to a deep copy of definition without regeneration."""
    updated = definition.model_copy(deep=True)

    if isinstance(patch, RetrievalPatch):
        if not patch.target_fields:
            raise ValueError("RetrievalPatch.target_fields cannot be empty.")
        unknown = set(patch.target_fields) - ALLOWED_RETRIEVAL_FIELDS
        if unknown:
            raise ValueError(f"Unsupported retrieval patch field(s): {sorted(unknown)}")
        ret_data = updated.retrieval.model_dump()
        ret_data.update(patch.target_fields)
        updated.retrieval = AgentRetrievalSettings.model_validate(ret_data)
        return updated

    if isinstance(patch, PromptPatch):
        text = patch.text.strip()
        if not text:
            raise ValueError("PromptPatch.text cannot be empty.")

        if patch.section == "extra_rules":
            rules = list(updated.system_prompt.extra_rules)
            if patch.op == "replace_rule":
                idx = patch.target_rule_index
                if idx is not None and 0 <= idx < len(rules):
                    rules[idx] = text
                elif rules:
                    rules[-1] = text
                else:
                    rules.append(text)
            else:
                formatted = f"Example: {text}" if patch.op == "add_example" else text
                if formatted not in rules:
                    rules.append(formatted)
            updated.system_prompt.extra_rules = rules
            return updated

        current = str(getattr(updated.system_prompt, patch.section)).strip()
        if patch.op == "replace_rule":
            lines = current.splitlines()
            if (
                patch.target_rule_index is not None
                and 0 <= patch.target_rule_index < len(lines)
            ):
                lines[patch.target_rule_index] = f"- {text}"
                new_val = "\n".join(lines)
            elif len(lines) > 1:
                lines[-1] = f"- {text}"
                new_val = "\n".join(lines)
            else:
                new_val = text
        elif patch.op == "add_example":
            addition = f"Example: {text}"
            new_val = current if addition in current else f"{current}\n- {addition}"
        else:
            new_val = current if text in current else f"{current}\n- {text}"

        setattr(updated.system_prompt, patch.section, new_val)
        return updated

    raise TypeError(f"Unsupported patch type: {type(patch)}")


def validate_retrieval_patch_scale(
    patch: RetrievalPatch,
    score_scale: dict[str, Any],
) -> tuple[RetrievalPatch | None, str | None]:
    """Validate/clamp RetrievalPatch.min_score against the retriever's normalized scale."""
    if "min_score" not in patch.target_fields:
        return patch, None
    raw_val = patch.target_fields["min_score"]
    if raw_val is None:
        return patch, None
    try:
        val = float(raw_val)
    except (TypeError, ValueError):
        return None, f"Invalid non-numeric min_score={raw_val!r} in RetrievalPatch."

    norm_min = float(score_scale.get("normalized_min", 0.0))
    norm_max = float(score_scale.get("normalized_max", 1.0))
    safe_max = float(score_scale.get("max_safe_min_score", 0.7))

    if val < norm_min or val > norm_max:
        return (
            None,
            f"RetrievalPatch min_score={val} outside normalized [{norm_min}, {norm_max}] range.",
        )

    if val > safe_max:
        updated_fields = dict(patch.target_fields)
        updated_fields["min_score"] = safe_max
        clamped = patch.model_copy(
            update={
                "target_fields": updated_fields,
                "rationale": f"{patch.rationale} (clamped min_score {val} -> {safe_max})".strip(),
            }
        )
        return clamped, None

    return patch, None


def validate_patch_operator_mapping(
    patches: list[OptimizerPatch],
    failures: list[ClassifiedFailure],
    score_scale: dict[str, Any] | None = None,
) -> tuple[list[OptimizerPatch], list[str]]:
    """Enforce failure-category -> operator mapping and retrieval scale constraints."""
    failure_by_id = {f.case_id: f for f in failures}
    present_cats = {f.failure_category for f in failures}
    scale = score_scale or {"normalized_min": 0.0, "normalized_max": 1.0, "max_safe_min_score": 0.7}

    valid: list[OptimizerPatch] = []
    rejections: list[str] = []

    for patch in patches:
        if isinstance(patch, RetrievalPatch):
            if "retrieval_miss" not in present_cats:
                rejections.append(
                    "Operator mapping rejected RetrievalPatch: no retrieval_miss failures present."
                )
                continue
            addressed_cats = {
                failure_by_id[cid].failure_category
                for cid in patch.addressed_case_ids
                if cid in failure_by_id
            }
            if addressed_cats and "retrieval_miss" not in addressed_cats:
                rejections.append(
                    f"Operator mapping rejected RetrievalPatch addressing {sorted(addressed_cats)} "
                    "(must target retrieval_miss only)."
                )
                continue
            scaled_patch, scale_err = validate_retrieval_patch_scale(patch, scale)
            if scaled_patch is None:
                rejections.append(scale_err or "Invalid retrieval min_score.")
                continue
            valid.append(scaled_patch)
            continue

        if isinstance(patch, PromptPatch):
            addressed_cats = {
                failure_by_id[cid].failure_category
                for cid in patch.addressed_case_ids
                if cid in failure_by_id
            }
            if addressed_cats and addressed_cats == {"retrieval_miss"}:
                rejections.append(
                    "Operator mapping rejected PromptPatch targeting only retrieval_miss failures."
                )
                continue
            if (
                addressed_cats == {"hallucination"}
                or (not addressed_cats and present_cats == {"hallucination"})
            ) and patch.section != "grounding_rules":
                rejections.append(
                    f"Operator mapping rejected PromptPatch on '{patch.section}' for "
                    "hallucination (must target 'grounding_rules')."
                )
                continue
            valid.append(patch)

    return valid, rejections


def apply_patches(
    definition: AgentDefinition, patches: list[OptimizerPatch]
) -> AgentDefinition:
    """Sequentially apply a list of typed patches to a copy of definition."""
    current = definition.model_copy(deep=True)
    for patch in patches:
        current = apply_patch(current, patch)
    return current
