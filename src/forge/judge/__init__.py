"""Automated two-layer evaluation judge and canary reliability self-test."""

from forge.judge.deterministic import (
    canonicalize_text,
    check_citation_validity,
    check_refusal,
    compute_key_fact_recall,
    find_ungrounded_numbers,
    match_key_fact,
    pearson_correlation,
    verify_quote_in_evidence,
    wilson_interval,
)
from forge.judge.models import (
    CanaryCaseResult,
    CanaryMetric,
    CaseVerdict,
    CategoryMetric,
    ClaimVerification,
    EvalCaseResult,
    EvalScorecard,
    JudgeReliabilityReport,
)
from forge.judge.scorer import Judge
from forge.judge.selftest import CanarySpec, build_canary_suite, run_judge_selftest

__all__ = [
    "CanaryCaseResult",
    "CanaryMetric",
    "CanarySpec",
    "CaseVerdict",
    "CategoryMetric",
    "ClaimVerification",
    "EvalCaseResult",
    "EvalScorecard",
    "Judge",
    "JudgeReliabilityReport",
    "build_canary_suite",
    "canonicalize_text",
    "check_citation_validity",
    "check_refusal",
    "compute_key_fact_recall",
    "find_ungrounded_numbers",
    "match_key_fact",
    "pearson_correlation",
    "run_judge_selftest",
    "verify_quote_in_evidence",
    "wilson_interval",
]
