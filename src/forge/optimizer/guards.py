"""Diff guard and eval-leak guard for safe targeted optimization."""

from __future__ import annotations

import json
import re

from forge.agent.schema import AgentDefinition
from forge.evals.models import EvalCase
from forge.optimizer.models import OptimizerPatch, PromptPatch, RetrievalPatch

_EVAL_ID_RE = re.compile(r"\b(?:DEV|HELD)-\d{3}\b")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


class DiffGuardViolation(AssertionError):
    """Raised when a patched AgentDefinition mutates fields outside allowed targets."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )


def verify_diff_guard(
    before: AgentDefinition,
    after: AgentDefinition,
    patches: list[OptimizerPatch],
) -> None:
    """Assert every field outside the patch targets is byte-identical."""
    allowed_retrieval: set[str] = set()
    allowed_sections: set[str] = set()
    for patch in patches:
        if isinstance(patch, RetrievalPatch):
            allowed_retrieval.update(patch.target_fields.keys())
        elif isinstance(patch, PromptPatch):
            allowed_sections.add(patch.section)

    before_d = before.model_dump(mode="json")
    after_d = after.model_dump(mode="json")

    for top_key in ("meta", "tools", "output_contract", "model"):
        if _canonical_bytes(before_d.get(top_key)) != _canonical_bytes(after_d.get(top_key)):
            raise DiffGuardViolation(
                f"Diff guard violation: top-level '{top_key}' changed unexpectedly."
            )

    b_ret = before_d.get("retrieval", {})
    a_ret = after_d.get("retrieval", {})
    for field in set(b_ret) | set(a_ret):
        if field not in allowed_retrieval and (
            _canonical_bytes(b_ret.get(field)) != _canonical_bytes(a_ret.get(field))
        ):
            raise DiffGuardViolation(
                f"Diff guard violation: retrieval.{field} mutated outside patch targets."
            )

    b_sys = before_d.get("system_prompt", {})
    a_sys = after_d.get("system_prompt", {})
    for sec in set(b_sys) | set(a_sys):
        if sec not in allowed_sections and _canonical_bytes(b_sys.get(sec)) != _canonical_bytes(
            a_sys.get(sec)
        ):
            raise DiffGuardViolation(
                f"Diff guard violation: system_prompt.{sec} mutated outside patch targets."
            )


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def check_patch_for_leak(
    patch: OptimizerPatch,
    all_eval_cases: list[EvalCase],
    *,
    ngram_size: int = 4,
    token_overlap_threshold: float = 0.70,
) -> tuple[bool, str | None]:
    """Return (is_clean, violation_reason) checking patch text against eval cases."""
    if not isinstance(patch, PromptPatch):
        return True, None

    patch_text = patch.text.strip()
    if _EVAL_ID_RE.search(patch_text):
        return False, "Patch text references explicit eval case IDs."

    patch_tokens = _norm_tokens(patch_text)
    patch_content = {t for t in patch_tokens if t not in _STOPWORDS and len(t) > 1}
    patch_ngrams = _ngrams(patch_tokens, ngram_size)
    patch_3grams = _ngrams(patch_tokens, 3)
    norm_patch_str = " ".join(patch_tokens)

    for case in all_eval_cases:
        targets: list[tuple[str, str]] = [("question", case.question)]
        for fact in case.key_facts:
            targets.append(("key_fact", fact))

        for kind, target_text in targets:
            t_tokens = _norm_tokens(target_text)
            if not t_tokens:
                continue
            norm_target_str = " ".join(t_tokens)

            if len(norm_target_str) >= 14 and norm_target_str in norm_patch_str:
                return False, (
                    f"Patch contains verbatim {kind} from eval case {case.id}: "
                    f"'{target_text[:60]}'"
                )

            t_ngrams = _ngrams(t_tokens, ngram_size)
            shared_4 = patch_ngrams & t_ngrams
            if shared_4:
                phrase = " ".join(next(iter(shared_4)))
                return False, (
                    f"Patch shares {ngram_size}-gram '{phrase}' with {kind} of {case.id}."
                )

            # For short key facts (3+ tokens), check 3-gram overlap too
            if kind == "key_fact" and len(t_tokens) >= 3:
                shared_3 = patch_3grams & _ngrams(t_tokens, 3)
                if shared_3:
                    phrase = " ".join(next(iter(shared_3)))
                    return False, (
                        f"Patch shares 3-gram '{phrase}' with key_fact of {case.id}."
                    )

            t_content = {t for t in t_tokens if t not in _STOPWORDS and len(t) > 1}
            if len(t_content) >= 4:
                overlap_ratio = len(patch_content & t_content) / len(t_content)
                if overlap_ratio >= token_overlap_threshold:
                    return False, (
                        f"Patch has high content overlap ({overlap_ratio:.2f}) with "
                        f"{kind} of {case.id}."
                    )

    return True, None


def filter_leaking_patches(
    patches: list[OptimizerPatch],
    dev_cases: list[EvalCase],
    held_out_cases: list[EvalCase],
) -> tuple[list[OptimizerPatch], list[str]]:
    """Filter out any patches triggering the leak guard across dev or held_out cases."""
    all_cases = [*dev_cases, *held_out_cases]
    clean: list[OptimizerPatch] = []
    rejections: list[str] = []
    for patch in patches:
        ok, reason = check_patch_for_leak(patch, all_cases)
        if ok:
            clean.append(patch)
        else:
            rejections.append(reason or "Leak guard rejected patch.")
    return clean, rejections
