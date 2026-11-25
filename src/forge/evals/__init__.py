"""Source-verified evaluation dataset generation and quality gates."""

from forge.evals.gates import (
    check_oos_unanswerable,
    check_solvability,
    dedupe_by_embedding,
    normalize_text,
    stratified_split,
    verify_case_key_facts,
    verify_key_fact,
)
from forge.evals.generator import EvalGenerator, default_category_quotas, model_family
from forge.evals.models import (
    EvalCase,
    EvalCategory,
    EvalProvenance,
    EvalSet,
    EvalSourceRef,
    EvalSplit,
    ExpectedBehavior,
    RejectedCaseLog,
)

__all__ = [
    "EvalCase",
    "EvalCategory",
    "EvalGenerator",
    "EvalProvenance",
    "EvalSet",
    "EvalSourceRef",
    "EvalSplit",
    "ExpectedBehavior",
    "RejectedCaseLog",
    "check_oos_unanswerable",
    "check_solvability",
    "dedupe_by_embedding",
    "default_category_quotas",
    "model_family",
    "normalize_text",
    "stratified_split",
    "verify_case_key_facts",
    "verify_key_fact",
]
