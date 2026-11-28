"""Code-templated honest quality report and scorecard.csv builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forge import __version__
from forge.agent.schema import AgentDefinition
from forge.config import load_prompt
from forge.evals.generator import model_family
from forge.evals.models import EvalSet
from forge.generator.synthesizer import GenerationLog
from forge.intake.models import RequirementsDocument
from forge.judge.models import EvalScorecard, JudgeReliabilityReport
from forge.llm.client import LLMClient
from forge.optimizer.models import OptimizationScorecard
from forge.report.lint import collect_allowed_numbers, lint_narrative_numbers


class _DiagnosisItem(BaseModel):
    case_ids: list[str] = Field(default_factory=list)
    diagnosis: str = ""


class _DiagnosisResponse(BaseModel):
    diagnosis_items: list[_DiagnosisItem] = Field(default_factory=list)
    overall_summary: str = ""


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_usage_summary(usage_path: Path) -> dict[str, Any]:
    if not usage_path.is_file():
        return {
            "calls": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "actual_providers_by_role": {},
            "actual_models_by_role": {},
            "role_model_counts": {},
        }
    calls = 0
    cache_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_latency_ms = 0.0
    actual_providers_ok: dict[str, list[str]] = {}
    actual_models_ok: dict[str, list[str]] = {}
    actual_providers_any: dict[str, list[str]] = {}
    actual_models_any: dict[str, list[str]] = {}
    role_model_counts: dict[str, dict[str, int]] = {}
    for line in usage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls += 1
        if rec.get("cached"):
            cache_hits += 1
        prompt_tokens += int(rec.get("prompt_tokens") or 0)
        completion_tokens += int(rec.get("completion_tokens") or 0)
        total_tokens += int(rec.get("total_tokens") or 0)
        total_latency_ms += float(rec.get("latency_ms") or 0.0)
        role = str(rec.get("role") or "").strip()
        prov = str(rec.get("provider") or "").strip()
        model = str(rec.get("model") or "").strip()
        status = str(rec.get("status") or "ok").strip()
        if role:
            if prov and prov not in actual_providers_any.setdefault(role, []):
                actual_providers_any[role].append(prov)
            if model and model not in actual_models_any.setdefault(role, []):
                actual_models_any[role].append(model)
            if status == "ok":
                if prov and prov not in actual_providers_ok.setdefault(role, []):
                    actual_providers_ok[role].append(prov)
                if model and model not in actual_models_ok.setdefault(role, []):
                    actual_models_ok[role].append(model)
            label = f"{prov}/{model}".strip("/")
            if label:
                counts = role_model_counts.setdefault(role, {})
                counts[label] = counts.get(label, 0) + 1

    actual_providers = {
        r: (actual_providers_ok.get(r) or actual_providers_any.get(r) or [])
        for r in set(actual_providers_any) | set(actual_providers_ok)
    }
    actual_models = {
        r: (actual_models_ok.get(r) or actual_models_any.get(r) or [])
        for r in set(actual_models_any) | set(actual_models_ok)
    }
    return {
        "calls": calls,
        "cache_hits": cache_hits,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_latency_ms": round(total_latency_ms, 2),
        "actual_providers_by_role": actual_providers,
        "actual_models_by_role": actual_models,
        "role_model_counts": role_model_counts,
    }


def _build_scorecard_csv(
    run_dir: Path,
    opt_scorecard: OptimizationScorecard | None,
    final_dev: EvalScorecard | None,
    final_held: EvalScorecard | None,
) -> Path:
    csv_path = run_dir / "scorecard.csv"
    rows: list[list[Any]] = []

    if opt_scorecard is not None:
        rows.append(
            [
                "baseline",
                "dev",
                "ALL",
                "",
                "",
                f"{opt_scorecard.baseline_dev_score:.4f}",
                "",
                "",
                "baseline dev score",
            ]
        )
        rows.append(
            [
                "baseline",
                "held_out",
                "ALL",
                "",
                "",
                f"{opt_scorecard.baseline_held_out_score:.4f}",
                "",
                "",
                "baseline held-out score",
            ]
        )
        for it in opt_scorecard.iterations:
            rows.append(
                [
                    f"iter_{it.iteration}",
                    "dev",
                    "ALL",
                    "",
                    "",
                    f"{it.score_after:.4f}",
                    "",
                    "",
                    f"accepted={it.accepted}; rolled_back={it.rolled_back}",
                ]
            )

    for label, sc in (("final_dev", final_dev), ("final_held_out", final_held)):
        if sc is None:
            continue
        rows.append(
            [
                label,
                sc.split,
                "ALL",
                sc.passed_cases,
                sc.total_cases,
                f"{sc.overall_pass_rate:.4f}",
                f"{sc.overall_wilson_low:.4f}",
                f"{sc.overall_wilson_high:.4f}",
                f"suspicion={sc.suspicion_flag}",
            ]
        )
        for m in sc.category_metrics:
            rows.append(
                [
                    label,
                    sc.split,
                    m.category,
                    m.passed,
                    m.total,
                    f"{m.pass_rate:.4f}",
                    f"{m.wilson_low:.4f}",
                    f"{m.wilson_high:.4f}",
                    "",
                ]
            )

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "stage",
                "split",
                "category",
                "passed",
                "total",
                "pass_rate",
                "wilson_low",
                "wilson_high",
                "notes",
            ]
        )
        writer.writerows(rows)
    return csv_path


def _generate_verified_diagnosis(
    failing_items: list[dict[str, Any]],
    allowed_numbers: set[str],
    summary_block: str,
    llm_client: LLMClient | None,
) -> tuple[list[str], list[str]]:
    """Generate and lint LLM diagnosis prose; fall back deterministically if lint rejects."""
    if not failing_items:
        return ["No failing cases remained in the final evaluation."], []

    valid_ids = {item["case_id"] for item in failing_items}
    lint_warnings: list[str] = []

    if llm_client is not None:
        failures_block = "\n".join(
            f"- {item['case_id']} ({item['split']}/{item['category']}, "
            f"failure={item['failure_category']}): {item['reasons']}"
            for item in failing_items
        )
        prompt = load_prompt("report_diagnosis.v1.md").format(
            summary_block=summary_block,
            allowed_numbers=sorted(allowed_numbers),
            failures_block=failures_block,
        )
        try:
            resp = llm_client.json(
                prompt,
                schema=_DiagnosisResponse,
                role="judge",
                temperature=0.0,
                max_repairs=1,
            )
            combined_parts = [resp.overall_summary] + [
                f"[{' '.join(d.case_ids)}] {d.diagnosis}" for d in resp.diagnosis_items
            ]
            combined_text = "\n".join(p for p in combined_parts if p.strip())
            ok, violations = lint_narrative_numbers(
                combined_text, allowed_numbers, valid_case_ids=valid_ids
            )
            if ok and resp.diagnosis_items:
                out_lines = [resp.overall_summary.strip()] if resp.overall_summary.strip() else []
                for item in resp.diagnosis_items:
                    cids = ", ".join(item.case_ids)
                    out_lines.append(f"- **[{cids}]**: {item.diagnosis.strip()}")
                return out_lines, []
            lint_warnings.extend(violations or ["LLM diagnosis omitted valid case IDs."])
        except Exception as exc:
            lint_warnings.append(f"LLM diagnosis generation failed: {exc}")

    # Deterministic code-templated fallback citing exact case IDs & verified result reasons
    fallback_lines = [
        "Deterministic failure diagnosis (all facts traced directly to result files):"
    ]
    for item in failing_items:
        fallback_lines.append(
            f"- **{item['case_id']}** (`{item['split']}` / `{item['category']}`, "
            f"`{item['failure_category']}`): {item['reasons']}"
        )
    return fallback_lines, lint_warnings


def build_report(
    run_dir: Path | str,
    *,
    llm_client: LLMClient | None = None,
    pipeline_state: dict[str, Any] | None = None,
    large_gap_threshold: float = 0.20,
) -> tuple[Path, Path]:
    """Build report.md and scorecard.csv strictly from persisted run result files."""
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)

    state = pipeline_state or _read_json_if_exists(target / "pipeline_state.json") or {}
    req_data = _read_json_if_exists(target / "requirements.json")
    gen_log_data = _read_json_if_exists(target / "generation_log.json")
    eval_set_data = _read_json_if_exists(target / "eval_set.json")
    opt_sc_data = _read_json_if_exists(target / "scorecard.json")
    final_dev_data = _read_json_if_exists(target / "final_dev_eval" / "results.json")
    final_held_data = _read_json_if_exists(target / "final_held_out_eval" / "results.json")
    judge_rel_data = _read_json_if_exists(target / "judge_reliability.json")
    usage_data = _read_usage_summary(target / "usage.jsonl")

    definition: AgentDefinition | None = None
    for def_cand in (target / "definition.yaml", target / "best_definition.yaml"):
        if def_cand.is_file():
            definition = AgentDefinition.load_yaml(def_cand)
            break

    req_doc = RequirementsDocument.model_validate(req_data) if req_data else None
    gen_log = GenerationLog.model_validate(gen_log_data) if gen_log_data else None
    eval_set = EvalSet.model_validate(eval_set_data) if eval_set_data else None
    opt_sc = OptimizationScorecard.model_validate(opt_sc_data) if opt_sc_data else None
    final_dev = EvalScorecard.model_validate(final_dev_data) if final_dev_data else None
    final_held = EvalScorecard.model_validate(final_held_data) if final_held_data else None
    judge_rel = (
        JudgeReliabilityReport.model_validate(judge_rel_data) if judge_rel_data else None
    )

    csv_path = _build_scorecard_csv(target, opt_sc, final_dev, final_held)

    # Collect allowed numbers from ALL persisted result files for narrative lint
    allowed_numbers = collect_allowed_numbers(
        {
            "state": state,
            "requirements": req_data,
            "generation_log": gen_log_data,
            "eval_set": eval_set_data,
            "scorecard": opt_sc_data,
            "final_dev": final_dev_data,
            "final_held": final_held_data,
            "judge_reliability": judge_rel_data,
            "usage": usage_data,
        }
    )

    lines: list[str] = ["# ForgeAI Run Quality Report", ""]

    # Partial failure banner if pipeline crashed mid-run
    if state.get("failed_stage"):
        lines.extend(
            [
                "## ⚠️ Partial Run Warning",
                "",
                f"- **Failed Stage**: `{state['failed_stage']}`",
                f"- **Error**: `{state.get('error', 'unknown error')}`",
                f"- **Completed Stages**: `{state.get('completed_stages', [])}`",
                "",
            ]
        )

    # Suspicion rule banner
    suspicion_reasons: list[str] = []
    if opt_sc is not None:
        if opt_sc.final_dev_score >= 1.0:
            suspicion_reasons.append(
                "Final dev pass rate is 100.0% — evaluation set may be too easy or overfit."
            )
        if eval_set and eval_set.held_out_cases and opt_sc.final_held_out_score >= 1.0:
            suspicion_reasons.append(
                "Final held-out pass rate is 100.0% — verify difficulty and canary coverage."
            )
        gap = opt_sc.final_dev_score - opt_sc.final_held_out_score
        if eval_set and eval_set.held_out_cases and abs(gap) >= large_gap_threshold:
            suspicion_reasons.append(
                f"Large dev vs held-out gap detected "
                f"(dev={opt_sc.final_dev_score * 100:.1f}%, "
                f"held_out={opt_sc.final_held_out_score * 100:.1f}%, "
                f"gap={gap * 100:.1f}pp)."
            )
    if suspicion_reasons:
        lines.extend(["## ⚠️ Suspicion Rule Triggered", ""])
        for s_reason in suspicion_reasons:
            lines.append(f"- {s_reason}")
        lines.append("")

    # 1. Run Summary (providers/models grounded in usage.jsonl actuals first)
    profile_val = state.get("profile") or (
        definition.meta.provenance.profile if definition else "default"
    )
    fallback_providers = (
        state.get("providers")
        or (definition.meta.provenance.providers if definition else None)
        or ({"agent": definition.model.provider} if definition else {})
    )
    fallback_models = (
        state.get("models")
        or (definition.meta.provenance.model_names if definition else None)
        or {}
    )
    # Prefer latest completed stage snapshot over stale initial state when usage.jsonl absent
    stage_models_state = state.get("stage_models") or {}
    if isinstance(stage_models_state, dict) and stage_models_state:
        latest_stage_map = list(stage_models_state.values())[-1]
        if isinstance(latest_stage_map, dict):
            for role_k, entry in latest_stage_map.items():
                if isinstance(entry, dict):
                    if entry.get("provider"):
                        fallback_providers = {**fallback_providers, role_k: entry["provider"]}
                    if entry.get("model"):
                        fallback_models = {**fallback_models, role_k: entry["model"]}

    actual_provs_by_role = usage_data.get("actual_providers_by_role", {})
    actual_mods_by_role = usage_data.get("actual_models_by_role", {})
    providers_map: dict[str, str] = {}
    models_map: dict[str, str] = {}
    for role_k in ("agent", "judge", "eval_writer", "embed"):
        provs = actual_provs_by_role.get(role_k) or []
        mods = actual_mods_by_role.get(role_k) or []
        if provs:
            providers_map[role_k] = provs[0] if len(provs) == 1 else " -> ".join(provs)
        elif role_k in fallback_providers:
            providers_map[role_k] = str(fallback_providers[role_k])
        if mods:
            models_map[role_k] = mods[0] if len(mods) == 1 else " -> ".join(mods)
        elif role_k in fallback_models:
            models_map[role_k] = str(fallback_models[role_k])

    providers_str = json.dumps(providers_map) if providers_map else "unknown"
    models_str = json.dumps(models_map) if models_map else "unknown"
    provenance_src = (
        "usage.jsonl actuals" if usage_data["calls"] > 0 else "checkpoint/definition metadata"
    )
    seed_val = (
        eval_set.seed
        if eval_set
        else (definition.model.seed if definition else "unknown")
    )
    lines.extend(
        [
            "## 1. Run Summary",
            f"- **Forge Version**: `{__version__}`",
            f"- **Agent Definition Version**: `{definition.meta.version if definition else 'n/a'}`",
            f"- **Profile**: `{profile_val}`",
            f"- **Providers**: `{providers_str}`",
            f"- **Models**: `{models_str}`",
            f"- **Model Provenance Source**: `{provenance_src}`",
            f"- **Seed**: `{seed_val}`",
            f"- **Fast Mode**: `{state.get('fast_mode', False)}`",
            f"- **Time Budget (min)**: `{state.get('time_budget_min', 'none')}` "
            f"(degraded=`{state.get('time_budget_degraded', False)}`)",
            f"- **Total Elapsed Seconds**: `{state.get('total_elapsed_seconds', 0.0)}`",
            f"- **Stage Timings (s)**: `{json.dumps(state.get('stage_timings_seconds', {}))}`",
            (
                f"- **Token Usage**: total=`{usage_data['total_tokens']}` "
                f"(prompt=`{usage_data['prompt_tokens']}`, "
                f"completion=`{usage_data['completion_tokens']}`, "
                f"calls=`{usage_data['calls']}`, cache_hits=`{usage_data['cache_hits']}`)"
            ),
        ]
    )
    if state.get("model_change_history"):
        lines.append("- **Model Changes During Resume**:")
        for change in state["model_change_history"]:
            lines.append(
                f"  - invalidated `{change.get('invalidated_stages', [])}`; "
                f"changed roles: `{json.dumps(change.get('changed_roles', {}))}`"
            )
    if state.get("time_budget_notes"):
        lines.append("- **Time Budget Degradation Notes**:")
        for note in state["time_budget_notes"]:
            lines.append(f"  - {note}")
    lines.append("")

    # 2. Requirements & Groundability Flags (shown up front)
    lines.append("## 2. Requirements & Groundability Flags")
    if req_doc is not None:
        lines.extend(
            [
                f"- **Brief**: {req_doc.brief}",
                f"- **Scope**: {req_doc.scope}",
                f"- **Audience / Tone**: {req_doc.audience} / {req_doc.tone}",
                (
                    f"- **Topics**: must_answer=`{len(req_doc.must_answer_topics)}`, "
                    f"must_refuse=`{len(req_doc.must_refuse_topics)}`, "
                    f"assumptions=`{len(req_doc.assumptions)}`"
                ),
            ]
        )
        rep = req_doc.groundability_report
        if rep is not None:
            lines.append(
                f"- **Groundability Summary**: `{rep.supported_count}` SUPPORTED, "
                f"`{rep.partial_count}` PARTIAL, `{rep.unsupported_count}` UNSUPPORTED"
            )
            if rep.flagged_gaps:
                lines.append("- **Flagged Groundability Gaps**:")
                for g in rep.flagged_gaps:
                    lines.append(f"  - {g}")
    else:
        lines.append("- Requirements artifact not produced.")
    lines.append("")

    # 3. Final Agent Definition Summary & Tools Attached vs Rejected
    lines.append("## 3. Final Agent Definition & Tool Gate Decisions")
    if definition is not None:
        lines.extend(
            [
                f"- **Agent Name**: `{definition.meta.name}`",
                (
                    f"- **Retrieval Config**: `chunk_size={definition.retrieval.chunk_size}`, "
                    f"`overlap={definition.retrieval.overlap}`, "
                    f"`top_k={definition.retrieval.top_k}`, "
                    f"`hybrid_alpha={definition.retrieval.hybrid_alpha}`, "
                    f"`rerank={definition.retrieval.rerank}`, "
                    f"`query_rewrite={definition.retrieval.query_rewrite}`, "
                    f"`min_score={definition.retrieval.min_score}`"
                ),
                f"- **Attached Tools**: `{definition.attached_tool_names() or ['none']}`",
            ]
        )
    if gen_log is not None and gen_log.tool_decisions:
        lines.append("- **Tool Gate Decisions**:")
        for td in gen_log.tool_decisions:
            status = "ATTACHED" if td.attached else "REJECTED"
            req_ref = td.justification_requirement_id or "-"
            lines.append(f"  - `{td.tool_name}` [{status}] (req=`{req_ref}`): {td.reason}")
    else:
        lines.append("- Tool gate log not available.")
    lines.append("")

    # 4. Per-Iteration Dev Scorecard & Final Held-Out Score with Wilson Intervals
    lines.append("## 4. Per-Iteration Dev Scorecard & Held-Out Results")
    if opt_sc is not None:
        lines.extend(
            [
                (
                    "- **Dev Score Trajectory**: "
                    f"baseline=`{opt_sc.baseline_dev_score * 100:.1f}%` "
                    f"→ final=`{opt_sc.final_dev_score * 100:.1f}%` "
                    f"(iterations=`{opt_sc.iterations_completed}`, stop=`{opt_sc.stop_reason}`)"
                ),
                (
                    f"- **Held-Out Score Trajectory**: "
                    f"baseline=`{opt_sc.baseline_held_out_score * 100:.1f}%` "
                    f"→ final=`{opt_sc.final_held_out_score * 100:.1f}%`"
                ),
                "",
                "| Iteration | Dev Before | Dev After | Accepted | Rolled Back | Decision |",
                "| --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for it in opt_sc.iterations:
            lines.append(
                f"| iter_{it.iteration} | {it.score_before * 100:.1f}% | "
                f"{it.score_after * 100:.1f}% | {it.accepted} | {it.rolled_back} | "
                f"{it.decision_reason} |"
            )
        lines.append("")

    if final_held is not None:
        lines.extend(
            [
                (
                    f"### Final Held-Out Score (Wilson 95% CI): "
                    f"`{final_held.passed_cases}/{final_held.total_cases}` "
                    f"(`{final_held.overall_pass_rate * 100:.1f}%`, "
                    f"[{final_held.overall_wilson_low * 100:.1f}%, "
                    f"{final_held.overall_wilson_high * 100:.1f}%])"
                ),
                "",
                "| Category | Passed | Total | Pass Rate | Wilson 95% CI |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for m in final_held.category_metrics:
            lines.append(
                f"| {m.category} | {m.passed} | {m.total} | "
                f"{m.pass_rate * 100:.1f}% | "
                f"[{m.wilson_low * 100:.1f}%, {m.wilson_high * 100:.1f}%] |"
            )
        lines.append("")
    elif final_dev is not None:
        lines.append(
            f"- Final dev scorecard available (`{final_dev.passed_cases}/{final_dev.total_cases}`, "
            f"`{final_dev.overall_pass_rate * 100:.1f}%`); held-out split not evaluated."
        )
        lines.append("")

    # 5. Failure-Category Counts per Iteration
    lines.append("## 5. Failure-Category Counts per Iteration")
    if opt_sc is not None and opt_sc.iterations:
        lines.extend(
            [
                "| Iteration | retrieval_miss | hallucination | wrong_refusal | prompt_defect |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for it in opt_sc.iterations:
            fc = it.failure_counts
            lines.append(
                f"| iter_{it.iteration} | {fc.get('retrieval_miss', 0)} | "
                f"{fc.get('hallucination', 0)} | {fc.get('wrong_refusal', 0)} | "
                f"{fc.get('prompt_defect', 0)} |"
            )
    else:
        lines.append("- No optimization iterations recorded.")
    lines.append("")

    # 6. What Still Fails and Why
    lines.append("## 6. What Still Fails and Why")
    failing_items: list[dict[str, Any]] = []
    for split_label, sc in (("held_out", final_held), ("dev", final_dev)):
        if sc is None:
            continue
        for res in sc.results:
            if not res.verdict.passed:
                is_contract_fail = res.answer_record.contract_failed or (
                    res.answer_record.output.refusal_reason
                    == "Model output failed output contract validation."
                )
                fail_cat = (
                    "prompt_defect (contract_failure)"
                    if is_contract_fail
                    else (
                        "wrong_refusal"
                        if not res.verdict.refusal_ok
                        else ("hallucination" if not res.verdict.grounded else "prompt_defect")
                    )
                )
                failing_items.append(
                    {
                        "case_id": res.case_id,
                        "split": split_label,
                        "category": res.category,
                        "failure_category": fail_cat,
                        "reasons": "; ".join(res.verdict.reasons),
                    }
                )

    summary_blk = (
        f"final_dev={final_dev.overall_pass_rate if final_dev else 0.0:.4f}, "
        f"final_held_out={final_held.overall_pass_rate if final_held else 0.0:.4f}"
    )
    diag_lines, lint_warnings = _generate_verified_diagnosis(
        failing_items, allowed_numbers, summary_blk, llm_client
    )
    lines.extend(diag_lines)
    if lint_warnings:
        lines.append("")
        lines.append(
            "- **Narrative Number Lint Note**: LLM diagnosis rejected ("
            + "; ".join(lint_warnings)
            + "); fell back to verified case-level facts."
        )
    lines.append("")

    # 7. Judge Reliability & Same-Model Warning
    lines.append("## 7. Judge Reliability & Bias Checks")
    if judge_rel is not None:
        lines.extend(
            [
                f"- **Judge Model**: `{judge_rel.judge_model}`",
                f"- **Canary Cases Evaluated**: `{judge_rel.total_canaries}`",
                f"- **Overall False-Pass Rate**: `{judge_rel.overall_false_pass_rate * 100:.1f}%`",
                f"- **Overall False-Fail Rate**: `{judge_rel.overall_false_fail_rate * 100:.1f}%`",
            ]
        )
    else:
        lines.append("- Judge self-test reliability report not available.")

    active_sc = final_held or final_dev
    agent_model_eff = (
        active_sc.agent_model
        if active_sc
        else (models_map.get("agent") or (definition.model.name if definition else ""))
    )
    judge_model_eff = (
        active_sc.judge_model
        if active_sc
        else (models_map.get("judge") or (judge_rel.judge_model if judge_rel else ""))
    )
    agent_prov_eff = providers_map.get("agent") or (definition.model.provider if definition else "")
    judge_prov_eff = providers_map.get("judge") or ""

    same_model_eff = (
        active_sc.same_model_as_agent
        if active_sc
        else bool(
            agent_model_eff
            and judge_model_eff
            and agent_model_eff.strip().lower() == judge_model_eff.strip().lower()
        )
    )
    same_family_eff = (
        active_sc.same_family_as_agent
        if active_sc
        else bool(
            agent_model_eff
            and judge_model_eff
            and model_family(agent_model_eff) == model_family(judge_model_eff)
        )
    )
    same_provider_eff = bool(
        agent_prov_eff
        and judge_prov_eff
        and agent_prov_eff.strip().lower() == judge_prov_eff.strip().lower()
    )

    if same_model_eff:
        lines.append(
            "- **⚠️ Same-Model Warning**: Agent and Judge use the exact same model "
            f"(`{agent_model_eff}`), risking self-preference bias."
        )
    if same_family_eff:
        lines.append(
            "- **⚠️ Same-Model-Family Warning**: Agent (`"
            f"{agent_model_eff}`) and Judge (`{judge_model_eff}`) share the same "
            "model family, which can introduce correlated bias."
        )
    if same_provider_eff:
        lines.append(
            "- **⚠️ Same-Provider Warning**: Agent and Judge both run on provider "
            f"`{agent_prov_eff}` (shared vendor/runtime stack)."
        )
    if agent_model_eff and judge_model_eff and not (same_model_eff or same_family_eff):
        lines.append(
            f"- **Model Separation Verified**: Agent (`{agent_model_eff}`) and Judge "
            f"(`{judge_model_eff}`) use distinct model families."
        )
    if active_sc and active_sc.length_pass_correlation is not None:
        lines.append(
            f"- **Answer Length vs Pass Correlation**: `{active_sc.length_pass_correlation:.3f}`"
        )
    lines.append("")

    # 8. Limitations
    lines.extend(
        [
            "## 8. Limitations",
            "- **Small Evaluation Sample Size**: Per-category dev and held-out splits are small; "
            "Wilson 95% confidence intervals reflect substantial sampling variance.",
            "- **Synthetic Evaluation Questions**: Eval cases are auto-generated from the corpus "
            "and may under-represent real user phrasing or unseen edge queries.",
            "- **Small Open-Model Constraints**: Target local models (3B–8B) remain sensitive to "
            "JSON schema drift, multi-hop synthesis limits, and imperfect tool selection.",
            "",
        ]
    )

    report_path = target / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, csv_path
