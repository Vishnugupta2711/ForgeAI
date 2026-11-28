"""Failure classification, typed patch operators, guards, and optimizer loop."""

from forge.optimizer.classifier import classify_failure, classify_failures
from forge.optimizer.guards import (
    DiffGuardViolation,
    check_patch_for_leak,
    filter_leaking_patches,
    verify_diff_guard,
)
from forge.optimizer.loop import OptimizerLoop
from forge.optimizer.models import (
    ClassifiedFailure,
    FailureCategory,
    IterationRecord,
    OptimizationScorecard,
    OptimizerPatch,
    PromptPatch,
    PromptPatchOp,
    PromptSectionName,
    RetrievalPatch,
)
from forge.optimizer.patches import (
    ALLOWED_RETRIEVAL_FIELDS,
    apply_patch,
    apply_patches,
    validate_patch_operator_mapping,
    validate_retrieval_patch_scale,
)

__all__ = [
    "ALLOWED_RETRIEVAL_FIELDS",
    "ClassifiedFailure",
    "DiffGuardViolation",
    "FailureCategory",
    "IterationRecord",
    "OptimizationScorecard",
    "OptimizerLoop",
    "OptimizerPatch",
    "PromptPatch",
    "PromptPatchOp",
    "PromptSectionName",
    "RetrievalPatch",
    "apply_patch",
    "apply_patches",
    "check_patch_for_leak",
    "classify_failure",
    "classify_failures",
    "filter_leaking_patches",
    "validate_patch_operator_mapping",
    "validate_retrieval_patch_scale",
    "verify_diff_guard",
]
