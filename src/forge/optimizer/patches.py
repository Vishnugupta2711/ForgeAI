"""Typed patch operators mutating only targeted AgentDefinition fields."""

from __future__ import annotations

from forge.agent.schema import AgentDefinition, AgentRetrievalSettings
from forge.optimizer.models import OptimizerPatch, PromptPatch, RetrievalPatch

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


def apply_patches(
    definition: AgentDefinition, patches: list[OptimizerPatch]
) -> AgentDefinition:
    """Sequentially apply a list of typed patches to a copy of definition."""
    current = definition.model_copy(deep=True)
    for patch in patches:
        current = apply_patch(current, patch)
    return current
