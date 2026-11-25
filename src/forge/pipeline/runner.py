"""Resumable end-to-end ForgeAI factory pipeline with stage checkpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

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
    fast_mode: bool = False
    strict: bool = False
    max_iters: int = 3
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
        stage_hooks: dict[str, Callable[[], None]] | None = None,
    ) -> PipelineState:
        """Execute the full resumable factory pipeline with partial report on failure."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        docs_path = Path(docs_dir).resolve()
        hooks = stage_hooks or {}
        t_start = time.perf_counter()

        resolved_max_iters = min(max_iters, 2) if fast else min(3, max(1, max_iters))
        state_file = out_path / "pipeline_state.json"

        if resume and state_file.is_file():
            state = PipelineState.model_validate_json(state_file.read_text(encoding="utf-8"))
            state.failed_stage = None
            state.error = None
        else:
            state = PipelineState(
                brief=brief,
                docs_dir=str(docs_path),
                out_dir=str(out_path),
                fast_mode=fast,
                strict=strict,
                max_iters=resolved_max_iters,
            )
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
                state.completed_stages.append("ingest")
                state.stage_timings_seconds["ingest"] = elapsed
                state.save(out_path)
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
                state.completed_stages.append("intake")
                state.stage_timings_seconds["intake"] = elapsed
                state.save(out_path)
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
                state.completed_stages.append("groundability")
                state.stage_timings_seconds["groundability"] = elapsed
                state.save(out_path)
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
                state.completed_stages.append("generate")
                state.stage_timings_seconds["generate"] = elapsed
                state.save(out_path)
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
                quotas = _fast_quotas(bool(definition.tools)) if fast else None
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
                state.completed_stages.append("evals")
                state.stage_timings_seconds["evals"] = elapsed
                state.save(out_path)
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
                run_judge_selftest(
                    self.llm_client,
                    chunk_map,
                    eval_set=eval_set,
                    config=self.config,
                    out_path=out_path / "judge_reliability.json",
                )
                elapsed = round(time.perf_counter() - s_t0, 3)
                state.completed_stages.append("judge_selftest")
                state.stage_timings_seconds["judge_selftest"] = elapsed
                state.save(out_path)
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
                opt = OptimizerLoop(
                    self.llm_client,
                    config=self.config,
                    max_iterations=resolved_max_iters,
                    bm25_only=self.bm25_only,
                )
                best_def, _ = opt.run(
                    definition,
                    eval_set,
                    docs_path,
                    out_dir=out_path,
                )
                # Keep <out_dir>/definition.yaml pointing to the final best definition
                best_def.save_yaml(out_path / "definition.yaml")
                elapsed = round(time.perf_counter() - s_t0, 3)
                state.completed_stages.append("optimize")
                state.stage_timings_seconds["optimize"] = elapsed
                state.save(out_path)
                self._emit("optimize", "done", elapsed)

            # 8. REPORT
            current_stage = "report"
            self._emit("report", "running", None)
            s_t0 = time.perf_counter()
            if "report" in hooks:
                hooks["report"]()
            state.total_elapsed_seconds = round(time.perf_counter() - t_start, 3)
            if "report" not in state.completed_stages:
                state.completed_stages.append("report")
            state.save(out_path)
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
    bm25_only: bool = False,
    llm_client: LLMClient | None = None,
) -> PipelineState:
    """Convenience entrypoint running ForgePipeline from configuration."""
    cfg = load_config()
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
    )
