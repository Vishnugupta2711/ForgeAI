"""Resumable end-to-end ForgeAI factory pipeline with stage checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from forge.agent.schema import AgentDefinition
from forge.config import ForgeConfig, load_config
from forge.evals import EvalCategory, EvalGenerator, EvalSet
from forge.generator import AgentGenerator
from forge.groundability import GroundabilityGuard
from forge.ingest import CorpusProfile, IngestResult, ingest_folder
from forge.intake import IntakeInterviewer, RequirementsDocument
from forge.judge import run_judge_selftest
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.optimizer import OptimizerLoop
from forge.report import build_report
from forge.retrieval import HybridIndex

STAGE_ORDER = [
    "ingest",
    "intake",
    "groundability",
    "generate",
    "evals",
    "judge_selftest",
    "optimize",
    "report",
]

ROLES: tuple[str, ...] = ("agent", "judge", "eval_writer", "embed")

STAGE_ROLE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ingest": ("embed",),
    "intake": ("agent",),
    "groundability": ("judge", "embed"),
    "generate": ("agent", "embed"),
    "evals": ("eval_writer", "embed", "agent"),
    "judge_selftest": ("judge", "embed"),
    "optimize": ("agent", "judge", "embed"),
    "report": ("judge",),
}


def _role_config_hash(provider: str, model: str) -> str:
    return hashlib.sha256(f"{provider}:{model}".encode()).hexdigest()[:16]


def compute_role_snapshot(
    cfg: ForgeConfig,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (providers_by_role, models_by_role, hashes_by_role) for current config."""
    providers = cfg.providers_by_role()
    models = {
        "agent": cfg.models.agent,
        "judge": cfg.models.judge,
        "eval_writer": cfg.models.eval_writer,
        "embed": cfg.models.embed,
    }
    hashes = {r: _role_config_hash(providers.get(r, ""), models.get(r, "")) for r in ROLES}
    return providers, models, hashes


class PipelineResumeError(RuntimeError):
    """Raised when --resume detects changed model/provider config without --allow-model-change."""


def _fast_quotas(has_tools: bool) -> dict[EvalCategory, int]:
    if has_tools:
        return {
            "single_doc": 4,
            "multi_doc": 2,
            "out_of_scope": 3,
            "edge_case": 2,
            "tool_use": 2,
        }
    return {
        "single_doc": 5,
        "multi_doc": 2,
        "out_of_scope": 4,
        "edge_case": 2,
        "tool_use": 0,
    }


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline stage fails after persisting state and partial report."""

    def __init__(self, stage: str, message: str, state: PipelineState) -> None:
        super().__init__(f"Stage '{stage}' failed: {message}")
        self.stage = stage
        self.state = state


class PipelineState(BaseModel):
    """Checkpoint manifest persisted to pipeline_state.json."""

    brief: str
    docs_dir: str
    out_dir: str
    profile: str = "default"
    providers: dict[str, str] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    role_hashes: dict[str, str] = Field(default_factory=dict)
    stage_role_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)
    stage_models: dict[str, dict[str, dict[str, str]]] = Field(default_factory=dict)
    model_change_history: list[dict[str, Any]] = Field(default_factory=list)
    fast_mode: bool = False
    strict: bool = False
    max_iters: int = 3
    time_budget_min: float | None = None
    time_budget_degraded: bool = False
    time_budget_notes: list[str] = Field(default_factory=list)
    completed_stages: list[str] = Field(default_factory=list)
    stage_timings_seconds: dict[str, float] = Field(default_factory=dict)
    failed_stage: str | None = None
    error: str | None = None
    total_elapsed_seconds: float = 0.0

    def save(self, out_dir: Path) -> Path:
        path = out_dir / "pipeline_state.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def _checkpoint_present(stage: str, out_dir: Path) -> bool:
    required: dict[str, list[Path]] = {
        "ingest": [
            out_dir / "profile.json",
            out_dir / "ingest_result.json",
            out_dir / "index" / "meta.json",
        ],
        "intake": [out_dir / "intake_raw.json"],
        "groundability": [out_dir / "requirements.json", out_dir / "requirements.md"],
        "generate": [out_dir / "definition.yaml", out_dir / "generation_log.json"],
        "evals": [out_dir / "eval_set.json", out_dir / "eval_set_stats.md"],
        "judge_selftest": [out_dir / "judge_reliability.json"],
        "optimize": [
            out_dir / "scorecard.json",
            out_dir / "best_definition.yaml",
            out_dir / "definition.yaml",
        ],
        "report": [out_dir / "report.md", out_dir / "scorecard.csv"],
    }
    return all(p.is_file() for p in required.get(stage, []))


class ForgePipeline:
    """Orchestrates ingest -> intake -> groundability -> generate -> evals -> optimize -> report."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        config: ForgeConfig | None = None,
        bm25_only: bool = False,
        progress_cb: Callable[[str, str, float | None], None] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or llm_client.config
        self.bm25_only = bm25_only
        self.progress_cb = progress_cb

    def _emit(self, stage: str, status: str, elapsed: float | None = None) -> None:
        if self.progress_cb is not None:
            self.progress_cb(stage, status, elapsed)

    def run(
        self,
        *,
        brief: str,
        docs_dir: Path | str,
        out_dir: Path | str,
        mode: Literal["interactive", "answers", "auto"] = "auto",
        answers_path: Path | None = None,
        input_fn: Callable[[str], str] | None = None,
        max_iters: int = 3,
        strict: bool = False,
        fast: bool = False,
        resume: bool = False,
        allow_model_change: bool = False,
        time_budget_min: float | None = None,
        stage_hooks: dict[str, Callable[[], None]] | None = None,
    ) -> PipelineState:
        """Execute the full resumable factory pipeline with partial report on failure."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        docs_path = Path(docs_dir).resolve()
        hooks = stage_hooks or {}
        t_start = time.perf_counter()
        deadline_mono = (
            time.monotonic() + (time_budget_min * 60.0)
            if time_budget_min is not None and time_budget_min > 0
            else None
        )

        resolved_max_iters = min(max_iters, 2) if fast else min(3, max(1, max_iters))
        state_file = out_path / "pipeline_state.json"
        new_providers, new_models, new_hashes = compute_role_snapshot(self.config)
        stage_snapshot = {
            r: {"provider": new_providers.get(r, ""), "model": new_models.get(r, "")}
            for r in ROLES
        }

        if resume and state_file.is_file():
            state = PipelineState.model_validate_json(state_file.read_text(encoding="utf-8"))
            old_hashes = dict(state.role_hashes)
            if not old_hashes:
                old_hashes = {
                    r: _role_config_hash(state.providers.get(r, ""), state.models.get(r, ""))
                    for r in ROLES
                }

            changed_roles: dict[str, dict[str, str]] = {}
            for r in ROLES:
                if old_hashes.get(r) != new_hashes[r]:
                    changed_roles[r] = {
                        "old_provider": state.providers.get(r, ""),
                        "old_model": state.models.get(r, ""),
                        "new_provider": new_providers.get(r, ""),
                        "new_model": new_models.get(r, ""),
                    }
            # Also check per-stage hashes in case an earlier completed stage used a different model
            for comp_stage in state.completed_stages:
                s_hashes = state.stage_role_hashes.get(comp_stage, {})
                s_models = state.stage_models.get(comp_stage, {})
                for r in STAGE_ROLE_DEPENDENCIES.get(comp_stage, ()):
                    if s_hashes.get(r) and s_hashes[r] != new_hashes[r] and r not in changed_roles:
                        changed_roles[r] = {
                            "old_provider": s_models.get(r, {}).get("provider", ""),
                            "old_model": s_models.get(r, {}).get("model", ""),
                            "new_provider": new_providers.get(r, ""),
                            "new_model": new_models.get(r, ""),
                        }

            if changed_roles:
                earliest_idx = len(STAGE_ORDER)
                for idx_s, s_name in enumerate(STAGE_ORDER):
                    deps = set(STAGE_ROLE_DEPENDENCIES.get(s_name, ()))
                    if deps & set(changed_roles):
                        earliest_idx = min(earliest_idx, idx_s)
                        break
                downstream = STAGE_ORDER[earliest_idx:]
                would_invalidate = [s for s in downstream if s in state.completed_stages]
                report_stages = would_invalidate or downstream
                if not allow_model_change:
                    role_details = "; ".join(
                        f"role '{r}' changed from {info['old_provider']}/{info['old_model']} "
                        f"to {info['new_provider']}/{info['new_model']}"
                        for r, info in sorted(changed_roles.items())
                    )
                    raise PipelineResumeError(
                        f"Refusing --resume because model config changed: {role_details}. "
                        f"Stages that would be invalidated: {report_stages}. "
                        "Pass --allow-model-change to invalidate and re-run dependent stages."
                    )
                invalidate_set = set(downstream)
                state.completed_stages = [
                    s for s in state.completed_stages if s not in invalidate_set
                ]
                for s_name in downstream:
                    state.stage_timings_seconds.pop(s_name, None)
                    state.stage_role_hashes.pop(s_name, None)
                    state.stage_models.pop(s_name, None)
                state.model_change_history.append(
                    {
                        "changed_roles": changed_roles,
                        "invalidated_stages": report_stages,
                    }
                )
                state.profile = self.config.profile
                state.providers = new_providers
                state.models = new_models
                state.role_hashes = new_hashes
                state.save(out_path)

            state.failed_stage = None
            state.error = None
            if time_budget_min is not None:
                state.time_budget_min = time_budget_min
        else:
            state = PipelineState(
                brief=brief,
                docs_dir=str(docs_path),
                out_dir=str(out_path),
                profile=self.config.profile,
                providers=new_providers,
                models=new_models,
                role_hashes=new_hashes,
                fast_mode=fast,
                strict=strict,
                max_iters=resolved_max_iters,
                time_budget_min=time_budget_min,
            )
            state.save(out_path)

        def _mark_stage_done(stage_name: str, elapsed_sec: float | None = None) -> None:
            if stage_name not in state.completed_stages:
                state.completed_stages.append(stage_name)
            if elapsed_sec is not None:
                state.stage_timings_seconds[stage_name] = elapsed_sec
            state.stage_role_hashes[stage_name] = dict(new_hashes)
            state.stage_models[stage_name] = {k: dict(v) for k, v in stage_snapshot.items()}
            state.save(out_path)

        def _should_skip(stage: str) -> bool:
            return (
                resume
                and stage in state.completed_stages
                and _checkpoint_present(stage, out_path)
            )

        ingest_res: IngestResult | None = None
        profile: CorpusProfile | None = None
        index: HybridIndex | None = None
        raw_req: RequirementsDocument | None = None
        verified_req: RequirementsDocument | None = None
        definition: AgentDefinition | None = None
        eval_set: EvalSet | None = None

        current_stage = "init"
        try:
            # 1. INGEST
            current_stage = "ingest"
            if _should_skip("ingest"):
                self._emit("ingest", "skipped (resumed)", 0.0)
                ingest_res = IngestResult.model_validate_json(
                    (out_path / "ingest_result.json").read_text(encoding="utf-8")
                )
                profile = CorpusProfile.model_validate_json(
                    (out_path / "profile.json").read_text(encoding="utf-8")
                )
                index = HybridIndex.load(out_path / "index")
            else:
                self._emit("ingest", "running", None)
                s_t0 = time.perf_counter()
                if "ingest" in hooks:
                    hooks["ingest"]()
                ingest_res, chunks, profile = ingest_folder(docs_path)
                (out_path / "ingest_result.json").write_text(
                    ingest_res.model_dump_json(indent=2), encoding="utf-8"
                )
                (out_path / "profile.json").write_text(
                    profile.model_dump_json(indent=2), encoding="utf-8"
                )
                if self.bm25_only:
                    index = HybridIndex.build(chunks, llm_client=None)
                else:
                    try:
                        index = HybridIndex.build(chunks, llm_client=self.llm_client)
                    except LLMProviderError:
                        index = HybridIndex.build(chunks, llm_client=None)
                index.save(out_path / "index")
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("ingest", elapsed)
                self._emit("ingest", "done", elapsed)

            assert ingest_res is not None and profile is not None and index is not None

            # 2. INTAKE
            current_stage = "intake"
            if _should_skip("intake"):
                self._emit("intake", "skipped (resumed)", 0.0)
                raw_req = RequirementsDocument.model_validate_json(
                    (out_path / "intake_raw.json").read_text(encoding="utf-8")
                )
            else:
                self._emit("intake", "running", None)
                s_t0 = time.perf_counter()
                if "intake" in hooks:
                    hooks["intake"]()
                raw_req = IntakeInterviewer(self.llm_client).run(
                    brief,
                    profile,
                    mode=mode,
                    answers_path=answers_path,
                    input_fn=input_fn,
                )
                (out_path / "intake_raw.json").write_text(
                    raw_req.model_dump_json(indent=2), encoding="utf-8"
                )
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("intake", elapsed)
                self._emit("intake", "done", elapsed)

            assert raw_req is not None

            # 3. GROUNDABILITY
            current_stage = "groundability"
            if _should_skip("groundability"):
                self._emit("groundability", "skipped (resumed)", 0.0)
                verified_req = RequirementsDocument.model_validate_json(
                    (out_path / "requirements.json").read_text(encoding="utf-8")
                )
            else:
                self._emit("groundability", "running", None)
                s_t0 = time.perf_counter()
                if "groundability" in hooks:
                    hooks["groundability"]()
                guard = GroundabilityGuard(index, self.llm_client)
                verified_req, _ = guard.evaluate(raw_req, strict=strict)
                verified_req.save(out_path)
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("groundability", elapsed)
                self._emit("groundability", "done", elapsed)

            assert verified_req is not None

            # 4. GENERATE
            current_stage = "generate"
            if _should_skip("generate"):
                self._emit("generate", "skipped (resumed)", 0.0)
                definition = AgentDefinition.load_yaml(out_path / "definition.yaml")
            else:
                self._emit("generate", "running", None)
                s_t0 = time.perf_counter()
                if "generate" in hooks:
                    hooks["generate"]()
                generator = AgentGenerator(self.llm_client, config=self.config)
                definition, _ = generator.generate(
                    verified_req,
                    profile,
                    index,
                    documents=ingest_res.documents,
                    out_dir=out_path,
                    run_smoke_test=not fast,
                )
                if fast and definition.retrieval.top_k > 4:
                    definition.retrieval.top_k = 4
                    definition.save_yaml(out_path / "definition.yaml")
                definition.save_yaml(out_path / "initial_definition.yaml")
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("generate", elapsed)
                self._emit("generate", "done", elapsed)

            assert definition is not None

            # 5. EVALS
            current_stage = "evals"
            if _should_skip("evals"):
                self._emit("evals", "skipped (resumed)", 0.0)
                eval_set = EvalSet.model_validate_json(
                    (out_path / "eval_set.json").read_text(encoding="utf-8")
                )
            else:
                self._emit("evals", "running", None)
                s_t0 = time.perf_counter()
                if "evals" in hooks:
                    hooks["evals"]()
                use_fast_evals = fast
                if deadline_mono is not None:
                    rem = deadline_mono - time.monotonic()
                    if rem < max(30.0, (time_budget_min or 0.0) * 60.0 * 0.5):
                        use_fast_evals = True
                        state.time_budget_degraded = True
                        state.time_budget_notes.append(
                            f"Before evals ({rem:.1f}s remaining): degraded to compact eval set."
                        )
                quotas = _fast_quotas(bool(definition.tools)) if use_fast_evals else None
                eval_gen = EvalGenerator(self.llm_client, config=self.config)
                eval_set = eval_gen.generate(
                    definition,
                    verified_req,
                    profile,
                    index,
                    quotas=quotas,
                    seed=self.config.generation.seed,
                    out_dir=out_path,
                )
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("evals", elapsed)
                self._emit("evals", "done", elapsed)

            assert eval_set is not None

            # 6. JUDGE SELFTEST
            current_stage = "judge_selftest"
            if _should_skip("judge_selftest"):
                self._emit("judge_selftest", "skipped (resumed)", 0.0)
            else:
                self._emit("judge_selftest", "running", None)
                s_t0 = time.perf_counter()
                if "judge_selftest" in hooks:
                    hooks["judge_selftest"]()
                chunk_map = {c.chunk_id: c for c in index.chunks}
                max_bases = 1 if (fast or state.time_budget_degraded) else 2
                run_judge_selftest(
                    self.llm_client,
                    chunk_map,
                    eval_set=eval_set,
                    config=self.config,
                    max_answer_bases=max_bases,
                    out_path=out_path / "judge_reliability.json",
                )
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("judge_selftest", elapsed)
                self._emit("judge_selftest", "done", elapsed)

            # 7. OPTIMIZE
            current_stage = "optimize"
            if _should_skip("optimize"):
                self._emit("optimize", "skipped (resumed)", 0.0)
            else:
                self._emit("optimize", "running", None)
                s_t0 = time.perf_counter()
                if "optimize" in hooks:
                    hooks["optimize"]()
                opt_iters = resolved_max_iters
                if deadline_mono is not None:
                    rem = deadline_mono - time.monotonic()
                    if rem <= 10.0:
                        opt_iters = 0
                        state.time_budget_degraded = True
                        state.time_budget_notes.append(
                            f"Before optimize ({rem:.1f}s remaining): "
                            "skipped optimization iterations."
                        )
                    elif rem < max(45.0, (time_budget_min or 0.0) * 60.0 * 0.4) and opt_iters > 1:
                        opt_iters = 1
                        state.time_budget_degraded = True
                        state.time_budget_notes.append(
                            f"Before optimize ({rem:.1f}s remaining): capped iterations to 1."
                        )
                opt = OptimizerLoop(
                    self.llm_client,
                    config=self.config,
                    max_iterations=opt_iters,
                    bm25_only=self.bm25_only,
                )
                best_def, _ = opt.run(
                    definition,
                    eval_set,
                    docs_path,
                    out_dir=out_path,
                    deadline_monotonic=deadline_mono,
                )
                # Keep <out_dir>/definition.yaml pointing to the final best definition
                best_def.save_yaml(out_path / "definition.yaml")
                elapsed = round(time.perf_counter() - s_t0, 3)
                _mark_stage_done("optimize", elapsed)
                self._emit("optimize", "done", elapsed)

            # 8. REPORT
            current_stage = "report"
            self._emit("report", "running", None)
            s_t0 = time.perf_counter()
            if "report" in hooks:
                hooks["report"]()
            state.total_elapsed_seconds = round(time.perf_counter() - t_start, 3)
            _mark_stage_done("report")
            build_report(
                out_path,
                llm_client=self.llm_client,
                pipeline_state=json.loads(state.model_dump_json()),
            )
            elapsed = round(time.perf_counter() - s_t0, 3)
            state.stage_timings_seconds["report"] = elapsed
            state.total_elapsed_seconds = round(time.perf_counter() - t_start, 3)
            state.save(out_path)
            # Re-write report with final report timing included
            build_report(
                out_path,
                llm_client=self.llm_client,
                pipeline_state=json.loads(state.model_dump_json()),
            )
            self._emit("report", "done", elapsed)
            return state

        except Exception as exc:
            state.failed_stage = current_stage
            state.error = str(exc)
            state.total_elapsed_seconds = round(time.perf_counter() - t_start, 3)
            state.save(out_path)
            build_report(
                out_path,
                llm_client=None,
                pipeline_state=json.loads(state.model_dump_json()),
            )
            self._emit(current_stage, f"failed: {exc}", state.total_elapsed_seconds)
            raise PipelineExecutionError(current_stage, str(exc), state) from exc


def run_factory(
    *,
    brief: str,
    docs_dir: Path | str,
    out_dir: Path | str,
    mode: Literal["interactive", "answers", "auto"] = "auto",
    answers_path: Path | None = None,
    max_iters: int = 3,
    strict: bool = False,
    fast: bool = False,
    resume: bool = False,
    allow_model_change: bool = False,
    time_budget_min: float | None = None,
    bm25_only: bool = False,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> PipelineState:
    """Convenience entrypoint running ForgePipeline from configuration."""
    cfg = load_config(profile=profile)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_path / "usage.jsonl")
    return ForgePipeline(client, config=cfg, bm25_only=bm25_only).run(
        brief=brief,
        docs_dir=docs_dir,
        out_dir=out_path,
        mode=mode,
        answers_path=answers_path,
        max_iters=max_iters,
        strict=strict,
        fast=fast,
        resume=resume,
        allow_model_change=allow_model_change,
        time_budget_min=time_budget_min,
    )
