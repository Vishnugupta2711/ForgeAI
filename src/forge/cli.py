"""Command-line interface for ForgeAI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge import __version__
from forge.agent.runtime import AgentRuntime
from forge.agent.schema import AgentDefinition, AnswerRecord
from forge.config import ForgeConfig, load_config
from forge.evals import EvalGenerator, EvalSet
from forge.generator import AgentGenerator, GenerationLog
from forge.groundability import GroundabilityGuard, StrictGroundabilityError
from forge.ingest.profile import ingest_folder
from forge.intake import IntakeInterviewer, RequirementsDocument
from forge.judge import EvalScorecard, Judge, JudgeReliabilityReport, run_judge_selftest
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.optimizer import OptimizationScorecard, OptimizerLoop
from forge.pipeline import ForgePipeline, PipelineExecutionError, PipelineResumeError, PipelineState
from forge.retrieval.hybrid import HybridIndex, HybridRetriever
from forge.retrieval.models import RetrievalConfig
from forge.tools import ToolContext

app = typer.Typer(
    name="forge",
    help="ForgeAI: autonomous factory for building, testing, and refining agents.",
    no_args_is_help=True,
)
evals_app = typer.Typer(
    name="evals",
    help="Generate and inspect source-verified evaluation datasets.",
    no_args_is_help=True,
)
app.add_typer(evals_app, name="evals")
console = Console()
_CLI_PROFILE: str | None = None


def _resolve_profile(profile: str | None) -> str | None:
    return profile if profile is not None else _CLI_PROFILE


def _load_cli_config(profile: str | None = None, *, warn_privacy: bool = True) -> ForgeConfig:
    try:
        cfg = load_config(profile=_resolve_profile(profile))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if warn_privacy:
        notice = cfg.privacy_notice()
        if notice:
            console.print(f"[yellow]{notice}[/yellow]")
    return cfg


@app.callback()
def main(
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """ForgeAI CLI root command."""
    global _CLI_PROFILE
    _CLI_PROFILE = profile


@app.command()
def version() -> None:
    """Show the installed ForgeAI version."""
    typer.echo(f"forge {__version__}")


@app.command()
def ingest(
    folder: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Document folder.")
    ],
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", min=64, help="Max characters per chunk.")
    ] = 800,
    overlap: Annotated[
        int, typer.Option("--overlap", min=0, help="Character overlap between chunks.")
    ] = 120,
    run_dir: Annotated[
        Path | None, typer.Option("--run-dir", help="Optional directory to save profile.json.")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print raw JSON corpus profile.")
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Ingest a document folder and print a compact corpus profile."""
    if profile is not None:
        _load_cli_config(profile, warn_privacy=False)
    _, _, corpus_profile = ingest_folder(folder, chunk_size=chunk_size, overlap=overlap)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "profile.json").write_text(
            corpus_profile.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if json_output:
        typer.echo(corpus_profile.model_dump_json(indent=2))
        return

    table = Table(title=f"ForgeAI Document Profile ({corpus_profile.root_path})")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Type")
    table.add_column("Size (B)", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Headings / Schema")

    for doc in corpus_profile.documents:
        if doc.csv_profile is not None:
            summary = (
                f"{doc.csv_profile.row_count} rows, "
                f"cols: {', '.join(doc.csv_profile.columns[:5])}"
            )
        elif doc.headings:
            summary = ", ".join(doc.headings[:4])
            if len(doc.headings) > 4:
                summary += f" (+{len(doc.headings) - 4} more)"
        else:
            summary = "-"
        table.add_row(
            doc.doc_id,
            doc.doc_type,
            str(doc.size_bytes),
            str(doc.page_count),
            str(doc.chunk_count),
            summary,
        )

    console.print(table)
    console.print(
        f"Scanned: {corpus_profile.total_files_scanned} | "
        f"Loaded: {corpus_profile.loaded_docs} | "
        f"Skipped: {corpus_profile.skipped_files} | "
        f"Total chunks: {corpus_profile.total_chunks}"
    )
    if corpus_profile.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in corpus_profile.warnings:
            console.print(f"  - {warning}")


@app.command()
def search(
    folder: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Document folder.")
    ],
    query: Annotated[str, typer.Argument(help="Search query string.")],
    top_k: Annotated[
        int, typer.Option("--top-k", min=1, help="Number of results to return.")
    ] = 5,
    alpha: Annotated[
        float,
        typer.Option("--alpha", min=0.0, max=1.0, help="Dense weight (0=BM25 only, 1=dense only)."),
    ] = 0.5,
    min_score: Annotated[
        float | None, typer.Option("--min-score", help="Optional minimum RRF score threshold.")
    ] = None,
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", min=64, help="Max characters per chunk.")
    ] = 800,
    overlap: Annotated[
        int, typer.Option("--overlap", min=0, help="Character overlap between chunks.")
    ] = 120,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Optional run directory to load/persist the hybrid index."),
    ] = None,
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Skip embedding calls and score with BM25 only."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Ingest/load a folder index and print ranked chunks with RRF scores."""
    cfg = _load_cli_config(profile, warn_privacy=not bm25_only)
    usage_log = (run_dir / "usage.jsonl") if run_dir is not None else None
    llm_client: LLMClient | None = (
        None if bm25_only else LLMClient.from_config(cfg, usage_log_path=usage_log)
    )

    index_dir = (run_dir / "index") if run_dir is not None else None
    index: HybridIndex | None = None
    if index_dir is not None and (index_dir / "meta.json").is_file():
        index = HybridIndex.load(index_dir)

    if index is None:
        _, chunks, profile = ingest_folder(folder, chunk_size=chunk_size, overlap=overlap)
        if profile.warnings:
            for warning in profile.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning}")
        try:
            index = HybridIndex.build(chunks, llm_client=llm_client)
        except LLMProviderError as exc:
            console.print(
                f"[yellow]Embedding provider unavailable ({exc}); "
                "falling back to BM25-only retrieval.[/yellow]"
            )
            llm_client = None
            index = HybridIndex.build(chunks, llm_client=None)

        if index_dir is not None:
            index.save(index_dir)

    effective_alpha = 0.0 if (llm_client is None or index.embeddings is None) else alpha
    retriever = HybridRetriever(
        index,
        llm_client=llm_client,
        config=RetrievalConfig(alpha=effective_alpha, top_k=top_k, min_score=min_score),
    )

    try:
        hits = retriever.search(query)
    except LLMProviderError as exc:
        console.print(
            f"[yellow]Embedding query failed ({exc}); retrying with BM25-only.[/yellow]"
        )
        hits = HybridRetriever(
            index,
            llm_client=None,
            config=RetrievalConfig(alpha=0.0, top_k=top_k, min_score=min_score),
        ).search(query)

    if not hits:
        console.print("[yellow]No matching chunks found.[/yellow]")
        return

    for rank, hit in enumerate(hits, start=1):
        loc_parts = [hit.chunk.doc_id]
        if hit.chunk.page is not None:
            loc_parts.append(f"p.{hit.chunk.page}")
        if hit.chunk.section_heading:
            loc_parts.append(hit.chunk.section_heading)
        header = (
            f"#{rank} | score={hit.score:.4f} | "
            f"bm25_rank={hit.bm25_rank or '-'} | dense_rank={hit.dense_rank or '-'} | "
            + " | ".join(loc_parts)
        )
        snippet = hit.chunk.text if len(hit.chunk.text) <= 400 else hit.chunk.text[:397] + "..."
        console.print(Panel(snippet, title=header, title_align="left"))


def _render_answer_record(record: AnswerRecord, *, show_trace: bool) -> None:
    out = record.output
    status = "[red]REFUSED[/red]" if out.refused else "[green]ANSWERED[/green]"
    title = f"{status} | confidence={out.confidence:.2f} | model={record.model}"
    body = out.answer
    if out.refused and out.refusal_reason:
        body = f"{out.answer}\n\n[dim]Refusal reason: {out.refusal_reason}[/dim]"
    console.print(Panel(body, title=title, title_align="left"))

    if out.citations:
        cit_table = Table(title="Citations")
        cit_table.add_column("Doc ID", style="cyan")
        cit_table.add_column("Chunk ID")
        cit_table.add_column("Quote")
        for cit in out.citations:
            cit_table.add_row(cit.doc_id, cit.chunk_id, cit.quote)
        console.print(cit_table)

    if show_trace:
        console.print("[bold]Trace:[/bold]")
        console.print(f"  effective_query: {record.effective_query}")
        if record.retrieved_chunks:
            console.print("  retrieved_chunks:")
            for rc in record.retrieved_chunks:
                console.print(
                    f"    - {rc.chunk_id} ({rc.doc_id}) score={rc.score:.4f} "
                    f"section={rc.section_heading or '-'}"
                )
        if record.tool_calls:
            console.print("  tool_calls:")
            for tc in record.tool_calls:
                console.print(
                    f"    - step={tc.step} {tc.tool_name}({tc.arguments}) "
                    f"ok={tc.ok} error={tc.error or '-'}"
                )
        for line in record.trace:
            console.print(f"  * {line}")


def _build_runtime_for_chat(
    definition: AgentDefinition,
    docs: Path,
    *,
    run_dir: Path | None,
    bm25_only: bool,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> AgentRuntime:
    cfg = _load_cli_config(profile, warn_privacy=False).model_copy(deep=True)
    cfg.endpoints.provider = definition.model.provider
    cfg.endpoints.role_providers["agent"] = definition.model.provider
    if definition.model.embed_provider is not None:
        cfg.endpoints.role_providers["embed"] = definition.model.embed_provider
    cfg.models.agent = definition.model.name
    cfg.models.embed = definition.model.embed_model
    cfg.generation.temperature = definition.model.temperature
    cfg.generation.seed = definition.model.seed
    if llm_client is None:
        notice = cfg.privacy_notice()
        if notice:
            console.print(f"[yellow]{notice}[/yellow]")

    usage_log = (run_dir / "usage.jsonl") if run_dir is not None else None
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=usage_log)

    ingest_res, chunks, _ = ingest_folder(
        docs,
        chunk_size=definition.retrieval.chunk_size,
        overlap=definition.retrieval.overlap,
    )

    index_dir = (run_dir / "index") if run_dir is not None else None
    index: HybridIndex | None = None
    if index_dir is not None and (index_dir / "meta.json").is_file():
        index = HybridIndex.load(index_dir)

    if index is None:
        if bm25_only:
            index = HybridIndex.build(chunks, llm_client=None)
        else:
            try:
                index = HybridIndex.build(
                    chunks,
                    llm_client=client,
                    embed_model=definition.model.embed_model,
                )
            except LLMProviderError as exc:
                console.print(
                    f"[yellow]Embedding provider unavailable ({exc}); "
                    "building BM25-only index.[/yellow]"
                )
                index = HybridIndex.build(chunks, llm_client=None)
        if index_dir is not None:
            index.save(index_dir)

    tool_cache_dir = (run_dir / "tool_cache") if run_dir is not None else None
    tool_ctx = ToolContext.from_documents(ingest_res.documents, cache_dir=tool_cache_dir)
    return AgentRuntime(definition, index, llm_client=client, tool_context=tool_ctx, config=cfg)


@app.command()
def chat(
    definition_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Path to agent definition YAML."),
    ],
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ],
    trace: Annotated[
        bool, typer.Option("--trace", help="Display retrieval and tool execution trace.")
    ] = False,
    question: Annotated[
        str | None,
        typer.Option("--question", "-q", help="Run a single question non-interactively."),
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Optional run directory for index/usage logs."),
    ] = None,
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build index without dense embedding calls."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Run an interactive multi-turn chat session using any declarative AgentDefinition."""
    definition = AgentDefinition.load_yaml(definition_path)
    resolved_prof = _resolve_profile(profile)
    extra_kwargs: dict[str, str] = {"profile": resolved_prof} if resolved_prof is not None else {}
    runtime = _build_runtime_for_chat(
        definition, docs, run_dir=run_dir, bm25_only=bm25_only, **extra_kwargs
    )

    console.print(
        f"[bold cyan]{definition.meta.name}[/bold cyan] v{definition.meta.version} "
        f"(tools: {definition.attached_tool_names() or ['none']})"
    )

    history: list[dict[str, str]] = []
    if question is not None:
        record = runtime.answer(question, history=history)
        _render_answer_record(record, show_trace=trace)
        return

    console.print("Type a question, or 'exit' / 'quit' to stop.")
    while True:
        try:
            if sys.stdin.isatty():
                user_input = console.input("[bold]You>[/bold] ")
            else:
                raw = sys.stdin.readline()
                if not raw:
                    break
                user_input = raw
        except (EOFError, KeyboardInterrupt):
            break

        cleaned = user_input.strip()
        if not cleaned:
            continue
        if cleaned.lower() in {"exit", "quit", ":q"}:
            break

        record = runtime.answer(cleaned, history=history)
        _render_answer_record(record, show_trace=trace)
        history.append({"role": "user", "content": cleaned})
        history.append({"role": "assistant", "content": record.output.answer})


def _run_intake_pipeline(
    *,
    brief_text: str,
    docs: Path,
    out_dir: Path,
    mode: str,
    answers_path: Path | None,
    strict: bool,
    bm25_only: bool,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> RequirementsDocument:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_dir / "usage.jsonl")

    _, chunks, profile = ingest_folder(docs)
    (out_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    if bm25_only:
        index = HybridIndex.build(chunks, llm_client=None)
    else:
        try:
            index = HybridIndex.build(chunks, llm_client=client)
        except LLMProviderError as exc:
            console.print(
                f"[yellow]Embedding provider unavailable ({exc}); "
                "building BM25-only index for groundability.[/yellow]"
            )
            index = HybridIndex.build(chunks, llm_client=None)
    index.save(out_dir / "index")

    interviewer = IntakeInterviewer(client)
    raw_req = interviewer.run(
        brief_text,
        profile,
        mode=mode,  # type: ignore[arg-type]
        answers_path=answers_path,
        input_fn=lambda q_prompt: console.input(f"[cyan]{q_prompt}[/cyan]"),
    )

    guard = GroundabilityGuard(index, client)
    try:
        verified_req, _ = guard.evaluate(raw_req, strict=strict)
    except StrictGroundabilityError as exc:
        if exc.requirements is not None:
            exc.requirements.save(out_dir)
        raise

    verified_req.save(out_dir)
    return verified_req


@app.command()
def intake(
    brief: Annotated[
        str,
        typer.Option("--brief", help="Path to a brief file or literal brief text."),
    ],
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ],
    auto: Annotated[
        bool,
        typer.Option("--auto", help="Simulate requester answers from brief + docs."),
    ] = False,
    answers: Annotated[
        Path | None,
        typer.Option("--answers", exists=True, dir_okay=False, help="YAML/JSON answers file."),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Abort if any topic has a groundability gap."),
    ] = False,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for requirements.json and requirements.md."),
    ] = Path("runs/intake"),
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build index without dense embedding calls."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Run clarifying intake and groundability guard to produce requirements.json/.md."""
    if auto and answers is not None:
        console.print("[red]Specify either --auto or --answers, not both.[/red]")
        raise typer.Exit(code=2)

    brief_path = Path(brief)
    brief_text = brief_path.read_text(encoding="utf-8").strip() if brief_path.is_file() else brief
    mode = "auto" if auto else ("answers" if answers is not None else "interactive")

    try:
        req = _run_intake_pipeline(
            brief_text=brief_text,
            docs=docs,
            out_dir=out,
            mode=mode,
            answers_path=answers,
            strict=strict,
            bm25_only=bm25_only,
            profile=profile,
        )
    except StrictGroundabilityError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    rep = req.groundability_report
    console.print(
        Panel(
            f"Scope: {req.scope}\n"
            f"Must answer: {len(req.must_answer_topics)} | "
            f"Must refuse: {len(req.must_refuse_topics)} | "
            f"Assumptions: {len(req.assumptions)}",
            title=f"Requirements Saved -> {out}",
            title_align="left",
        )
    )
    if rep is not None:
        console.print(
            f"Groundability: {rep.supported_count} SUPPORTED, "
            f"{rep.partial_count} PARTIAL, {rep.unsupported_count} UNSUPPORTED"
        )
        if rep.has_gaps:
            console.print("[yellow]Flagged groundability gaps:[/yellow]")
            for gap in rep.flagged_gaps:
                console.print(f"  - {gap}")


def _run_generate_pipeline(
    *,
    requirements_path: Path,
    docs: Path,
    out_dir: Path,
    agent_name: str | None,
    bm25_only: bool,
    run_smoke: bool,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> tuple[AgentDefinition, GenerationLog]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_dir / "usage.jsonl")
    requirements = RequirementsDocument.model_validate_json(
        requirements_path.read_text(encoding="utf-8")
    )

    ingest_res, chunks, profile = ingest_folder(docs)
    (out_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    index_dir = out_dir / "index"
    index: HybridIndex | None = None
    if (index_dir / "meta.json").is_file():
        index = HybridIndex.load(index_dir)

    if index is None:
        if bm25_only:
            index = HybridIndex.build(chunks, llm_client=None)
        else:
            try:
                index = HybridIndex.build(chunks, llm_client=client)
            except LLMProviderError as exc:
                console.print(
                    f"[yellow]Embedding provider unavailable ({exc}); "
                    "building BM25-only index.[/yellow]"
                )
                index = HybridIndex.build(chunks, llm_client=None)
        index.save(index_dir)

    generator = AgentGenerator(client, config=cfg)
    return generator.generate(
        requirements,
        profile,
        index,
        documents=ingest_res.documents,
        agent_name=agent_name,
        out_dir=out_dir,
        run_smoke_test=run_smoke,
    )


@app.command()
def generate(
    requirements: Annotated[
        Path,
        typer.Option(
            "--requirements",
            exists=True,
            dir_okay=False,
            help="Path to requirements.json produced by forge intake.",
        ),
    ],
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for definition.yaml and generation_log.json."),
    ] = Path("runs/generated"),
    name: Annotated[
        str | None,
        typer.Option("--name", help="Optional override for generated agent name."),
    ] = None,
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build index without dense embedding calls."),
    ] = False,
    no_smoke: Annotated[
        bool,
        typer.Option("--no-smoke", help="Skip post-generation 3-question smoke test."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Generate definition.yaml and generation_log.json from requirements and docs."""
    definition, gen_log = _run_generate_pipeline(
        requirements_path=requirements,
        docs=docs,
        out_dir=out,
        agent_name=name,
        bm25_only=bm25_only,
        run_smoke=not no_smoke,
        profile=profile,
    )
    attached = definition.attached_tool_names()
    console.print(
        Panel(
            f"Agent: {definition.meta.name}\n"
            f"Tools attached: {attached or ['none']}\n"
            f"Retrieval: chunk_size={definition.retrieval.chunk_size}, "
            f"overlap={definition.retrieval.overlap}, top_k={definition.retrieval.top_k}, "
            f"alpha={definition.retrieval.hybrid_alpha}\n"
            f"Smoke tests run: {len(gen_log.smoke_test_results)}",
            title=f"Generated Definition -> {out / 'definition.yaml'}",
            title_align="left",
        )
    )


def _run_evals_generate_pipeline(
    *,
    definition_path: Path,
    requirements_path: Path,
    docs: Path,
    out_dir: Path,
    seed: int | None,
    bm25_only: bool,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> EvalSet:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_dir / "usage.jsonl")
    definition = AgentDefinition.load_yaml(definition_path)
    requirements = RequirementsDocument.model_validate_json(
        requirements_path.read_text(encoding="utf-8")
    )

    _, chunks, profile = ingest_folder(
        docs,
        chunk_size=definition.retrieval.chunk_size,
        overlap=definition.retrieval.overlap,
    )
    index_dir = out_dir / "index"
    index: HybridIndex | None = None
    if (index_dir / "meta.json").is_file():
        index = HybridIndex.load(index_dir)

    if index is None:
        if bm25_only:
            index = HybridIndex.build(chunks, llm_client=None)
        else:
            try:
                index = HybridIndex.build(
                    chunks,
                    llm_client=client,
                    embed_model=definition.model.embed_model,
                )
            except LLMProviderError as exc:
                console.print(
                    f"[yellow]Embedding provider unavailable ({exc}); "
                    "building BM25-only index.[/yellow]"
                )
                index = HybridIndex.build(chunks, llm_client=None)
        index.save(index_dir)

    generator = EvalGenerator(client, config=cfg)
    return generator.generate(
        definition,
        requirements,
        profile,
        index,
        seed=seed,
        out_dir=out_dir,
    )


@evals_app.command("generate")
def evals_generate(
    definition: Annotated[
        Path,
        typer.Option(
            "--definition",
            exists=True,
            dir_okay=False,
            help="Path to definition.yaml.",
        ),
    ],
    requirements: Annotated[
        Path,
        typer.Option(
            "--requirements",
            exists=True,
            dir_okay=False,
            help="Path to requirements.json.",
        ),
    ],
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for eval_set.json and eval_set_stats.md."),
    ] = Path("runs/evals"),
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for stratified dev/held_out split."),
    ] = None,
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build index without dense embedding calls."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Generate a verified dev/held_out evaluation set from corpus documents."""
    eval_set = _run_evals_generate_pipeline(
        definition_path=definition,
        requirements_path=requirements,
        docs=docs,
        out_dir=out,
        seed=seed,
        bm25_only=bm25_only,
        profile=profile,
    )
    console.print(
        Panel(
            f"Total verified cases: {eval_set.total_cases} "
            f"(dev={len(eval_set.dev_cases)}, held_out={len(eval_set.held_out_cases)})\n"
            f"Rejected candidates: {len(eval_set.rejected_cases)}\n"
            f"Eval writer model: {eval_set.eval_writer_model} "
            f"(same_family_as_agent={eval_set.same_family_as_agent})\n"
            f"Categories: {eval_set.category_counts}",
            title=f"Eval Set Saved -> {out / 'eval_set.json'}",
            title_align="left",
        )
    )


def _run_evals_suite_pipeline(
    *,
    definition_path: Path,
    eval_set_path: Path,
    docs: Path,
    split: str,
    out_dir: Path,
    bm25_only: bool,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> EvalScorecard:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_dir / "usage.jsonl")
    definition = AgentDefinition.load_yaml(definition_path)
    eval_set = EvalSet.model_validate_json(eval_set_path.read_text(encoding="utf-8"))

    if split == "dev":
        cases = list(eval_set.dev_cases)
    elif split == "held_out":
        cases = list(eval_set.held_out_cases)
    elif split == "all":
        cases = [*eval_set.dev_cases, *eval_set.held_out_cases]
    else:
        raise ValueError(f"Unsupported split '{split}'. Use dev, held_out, or all.")

    ingest_res, chunks, _ = ingest_folder(
        docs,
        chunk_size=definition.retrieval.chunk_size,
        overlap=definition.retrieval.overlap,
    )
    chunk_map = {c.chunk_id: c for c in chunks}

    index_dir = out_dir / "index"
    index: HybridIndex | None = None
    if (index_dir / "meta.json").is_file():
        index = HybridIndex.load(index_dir)

    if index is None:
        if bm25_only:
            index = HybridIndex.build(chunks, llm_client=None)
        else:
            try:
                index = HybridIndex.build(
                    chunks, llm_client=client, embed_model=definition.model.embed_model
                )
            except LLMProviderError as exc:
                console.print(
                    f"[yellow]Embedding provider unavailable ({exc}); "
                    "building BM25-only index.[/yellow]"
                )
                index = HybridIndex.build(chunks, llm_client=None)
        index.save(index_dir)

    tool_ctx = ToolContext.from_documents(
        ingest_res.documents,
        cache_dir=out_dir / "tool_cache",
        offline_only=True,
    )
    runtime = AgentRuntime(
        definition, index, llm_client=client, tool_context=tool_ctx, config=cfg
    )
    records = [runtime.answer(c.question) for c in cases]

    judge = Judge(client, config=cfg, seed=eval_set.seed)
    return judge.evaluate_suite(
        definition,
        cases,
        records,
        chunk_map,
        split=split,
        out_dir=out_dir,
    )


@evals_app.command("run")
def evals_run(
    definition: Annotated[
        Path,
        typer.Option("--definition", exists=True, dir_okay=False, help="Path to definition.yaml."),
    ],
    eval_set: Annotated[
        Path,
        typer.Option("--eval-set", exists=True, dir_okay=False, help="Path to eval_set.json."),
    ],
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="Which eval split to run: dev, held_out, or all."),
    ] = "dev",
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for results.json and scorecard.md."),
    ] = Path("runs/eval_run"),
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build index without dense embedding calls."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Run an agent definition against an eval set and score with ForgeAI Judge."""
    scorecard = _run_evals_suite_pipeline(
        definition_path=definition,
        eval_set_path=eval_set,
        docs=docs,
        split=split,
        out_dir=out,
        bm25_only=bm25_only,
        profile=profile,
    )
    console.print(
        Panel(
            f"Split: {scorecard.split} | Passed: {scorecard.passed_cases}/{scorecard.total_cases} "
            f"({scorecard.overall_pass_rate * 100:.1f}%)\n"
            f"Wilson 95% CI: [{scorecard.overall_wilson_low * 100:.1f}%, "
            f"{scorecard.overall_wilson_high * 100:.1f}%]\n"
            f"Judge: {scorecard.judge_model} (same_family={scorecard.same_family_as_agent})",
            title=f"Scorecard Saved -> {out / 'scorecard.md'}",
            title_align="left",
        )
    )


def _run_judge_selftest_pipeline(
    *,
    docs: Path,
    eval_set_path: Path | None,
    out_dir: Path,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> JudgeReliabilityReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_dir / "usage.jsonl")
    _, chunks, _ = ingest_folder(docs)
    chunk_map = {c.chunk_id: c for c in chunks}
    eval_set = (
        EvalSet.model_validate_json(eval_set_path.read_text(encoding="utf-8"))
        if eval_set_path is not None
        else None
    )
    return run_judge_selftest(
        client,
        chunk_map,
        eval_set=eval_set,
        config=cfg,
        out_path=out_dir / "judge_reliability.json",
    )


@app.command("judge-selftest")
def judge_selftest(
    eval_set: Annotated[
        Path | None,
        typer.Option("--eval-set", exists=True, dir_okay=False, help="Optional eval_set.json."),
    ] = None,
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ] = Path("fixtures/docs"),
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for judge_reliability.json."),
    ] = Path("runs/judge_selftest"),
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Run canary reliability self-test for the ForgeAI Judge."""
    report = _run_judge_selftest_pipeline(
        docs=docs, eval_set_path=eval_set, out_dir=out, profile=profile
    )
    table = Table(title=f"Judge Self-Test Reliability — Standard Tier ({report.judge_model})")
    table.add_column("Canary Type", style="cyan")
    table.add_column("Expected")
    table.add_column("Total", justify="right")
    table.add_column("Clean Pass", justify="right")
    table.add_column("Fallback Pass", justify="right")
    table.add_column("False Pass Rate", justify="right")
    table.add_column("False Fail Rate", justify="right")
    for m in report.per_canary_type:
        table.add_row(
            m.canary_type,
            "PASS" if m.expected_pass else "FAIL",
            str(m.total),
            str(m.passed_cleanly),
            str(m.passed_via_fallback),
            f"{m.false_pass_rate * 100:.1f}%",
            f"{m.false_fail_rate * 100:.1f}%",
        )
    console.print(table)
    if report.hard_tier_per_canary_type:
        hard_table = Table(
            title=f"Judge Self-Test Reliability — HARD Tier ({report.judge_model})"
        )
        hard_table.add_column("Canary Type", style="magenta")
        hard_table.add_column("Expected")
        hard_table.add_column("Total", justify="right")
        hard_table.add_column("Clean Pass", justify="right")
        hard_table.add_column("Fallback Pass", justify="right")
        hard_table.add_column("False Pass Rate", justify="right")
        hard_table.add_column("False Fail Rate", justify="right")
        for m in report.hard_tier_per_canary_type:
            hard_table.add_row(
                m.canary_type,
                "PASS" if m.expected_pass else "FAIL",
                str(m.total),
                str(m.passed_cleanly),
                str(m.passed_via_fallback),
                f"{m.false_pass_rate * 100:.1f}%",
                f"{m.false_fail_rate * 100:.1f}%",
            )
        console.print(hard_table)
    console.print(
        f"Saved -> {out / 'judge_reliability.json'} | "
        f"Standard False Pass: {report.overall_false_pass_rate * 100:.1f}% | "
        f"Standard False Fail: {report.overall_false_fail_rate * 100:.1f}% | "
        f"Hard False Pass: {report.hard_tier_false_pass_rate * 100:.1f}% | "
        f"Hard False Fail: {report.hard_tier_false_fail_rate * 100:.1f}%"
    )


def _run_optimize_pipeline(
    *,
    run_dir: Path,
    docs: Path | None,
    max_iters: int,
    bm25_only: bool,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> OptimizationScorecard:
    def_path = run_dir / "definition.yaml"
    eval_path = run_dir / "eval_set.json"
    if not def_path.is_file():
        raise FileNotFoundError(f"Missing definition.yaml in {run_dir}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing eval_set.json in {run_dir}")

    resolved_docs = docs
    if resolved_docs is None:
        prof_path = run_dir / "profile.json"
        if prof_path.is_file():
            prof_data = json.loads(prof_path.read_text(encoding="utf-8"))
            if root_str := prof_data.get("root_path"):
                resolved_docs = Path(root_str)
    if resolved_docs is None or not resolved_docs.is_dir():
        raise FileNotFoundError(
            "Could not resolve --docs directory from arguments or profile.json."
        )

    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=run_dir / "usage.jsonl")
    definition = AgentDefinition.load_yaml(def_path)
    eval_set = EvalSet.model_validate_json(eval_path.read_text(encoding="utf-8"))

    loop = OptimizerLoop(
        client,
        config=cfg,
        max_iterations=max_iters,
        bm25_only=bm25_only,
    )
    _, opt_scorecard = loop.run(definition, eval_set, resolved_docs, out_dir=run_dir)
    return opt_scorecard


@app.command()
def optimize(
    run: Annotated[
        Path,
        typer.Option(
            "--run",
            exists=True,
            file_okay=False,
            help="Run directory containing definition.yaml and eval_set.json.",
        ),
    ],
    docs: Annotated[
        Path | None,
        typer.Option(
            "--docs", exists=True, file_okay=False, help="Optional document folder override."
        ),
    ] = None,
    max_iters: Annotated[
        int,
        typer.Option("--max-iters", min=1, max=3, help="Maximum optimization iterations (max 3)."),
    ] = 3,
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build retrieval index without dense embeddings."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Run targeted critic-optimizer loop on dev evals and score held-out once at the end."""
    opt_scorecard = _run_optimize_pipeline(
        run_dir=run,
        docs=docs,
        max_iters=max_iters,
        bm25_only=bm25_only,
        profile=profile,
    )
    console.print(
        Panel(
            f"Dev score: {opt_scorecard.baseline_dev_score * 100:.1f}% -> "
            f"{opt_scorecard.final_dev_score * 100:.1f}%\n"
            f"Held-out score: {opt_scorecard.baseline_held_out_score * 100:.1f}% -> "
            f"{opt_scorecard.final_held_out_score * 100:.1f}%\n"
            f"Iterations: {opt_scorecard.iterations_completed} | "
            f"Stop reason: {opt_scorecard.stop_reason}",
            title=f"Optimization Complete -> {run / 'scorecard.json'}",
            title_align="left",
        )
    )


def _run_full_pipeline_cli(
    *,
    brief_text: str,
    docs: Path,
    out_dir: Path,
    mode: str,
    answers_path: Path | None,
    max_iters: int,
    strict: bool,
    fast: bool,
    resume: bool,
    allow_model_change: bool = False,
    time_budget_min: float | None = None,
    bm25_only: bool = False,
    profile: str | None = None,
    llm_client: LLMClient | None = None,
) -> PipelineState:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cli_config(profile)
    client = llm_client or LLMClient.from_config(cfg, usage_log_path=out_dir / "usage.jsonl")

    def _progress(stage: str, status: str, elapsed: float | None) -> None:
        if elapsed is None:
            console.print(f"[cyan][{stage}][/cyan] {status}...")
        else:
            console.print(f"[green][{stage}][/green] {status} ({elapsed:.2f}s)")

    pipeline = ForgePipeline(
        client, config=cfg, bm25_only=bm25_only, progress_cb=_progress
    )
    return pipeline.run(
        brief=brief_text,
        docs_dir=docs,
        out_dir=out_dir,
        mode=mode,  # type: ignore[arg-type]
        answers_path=answers_path,
        input_fn=lambda q_prompt: console.input(f"[cyan]{q_prompt}[/cyan]"),
        max_iters=max_iters,
        strict=strict,
        fast=fast,
        resume=resume,
        allow_model_change=allow_model_change,
        time_budget_min=time_budget_min,
    )


@app.command()
def run(
    brief: Annotated[
        str,
        typer.Option("--brief", help="Path to a brief file or literal brief text."),
    ],
    docs: Annotated[
        Path,
        typer.Option("--docs", exists=True, file_okay=False, help="Document folder."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output run directory."),
    ] = Path("runs/latest"),
    auto: Annotated[
        bool,
        typer.Option("--auto", help="Run intake unattended with simulated requester."),
    ] = False,
    answers: Annotated[
        Path | None,
        typer.Option("--answers", exists=True, dir_okay=False, help="YAML/JSON answers file."),
    ] = None,
    max_iters: Annotated[
        int,
        typer.Option("--max-iters", min=1, max=3, help="Max optimizer iterations."),
    ] = 3,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Abort if groundability guard finds any gaps."),
    ] = False,
    fast: Annotated[
        bool,
        typer.Option("--fast", help="Use smaller eval set and lower retrieval/iter budgets."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume from completed stage checkpoints in --out."),
    ] = False,
    allow_model_change: Annotated[
        bool,
        typer.Option(
            "--allow-model-change",
            help=(
                "Allow --resume when model/provider config changed by "
                "invalidating downstream stages."
            ),
        ),
    ] = False,
    time_budget_min: Annotated[
        float | None,
        typer.Option(
            "--time-budget-min",
            min=0.1,
            help=(
                "Optional soft time budget in minutes; "
                "degrades eval/optimizer budgets gracefully."
            ),
        ),
    ] = None,
    bm25_only: Annotated[
        bool,
        typer.Option("--bm25-only", help="Build retrieval index without dense embeddings."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile (e.g. 'default', 'gemini')."),
    ] = None,
) -> None:
    """Run the full ForgeAI factory end-to-end and write report.md + scorecard.csv."""
    if auto and answers is not None:
        console.print("[red]Specify either --auto or --answers, not both.[/red]")
        raise typer.Exit(code=2)

    brief_path = Path(brief)
    brief_text = brief_path.read_text(encoding="utf-8").strip() if brief_path.is_file() else brief
    mode = "auto" if auto else ("answers" if answers is not None else "interactive")

    try:
        state = _run_full_pipeline_cli(
            brief_text=brief_text,
            docs=docs,
            out_dir=out,
            mode=mode,
            answers_path=answers,
            max_iters=max_iters,
            strict=strict,
            fast=fast,
            resume=resume,
            allow_model_change=allow_model_change,
            time_budget_min=time_budget_min,
            bm25_only=bm25_only,
            profile=profile,
        )
    except PipelineResumeError as exc:
        console.print(
            Panel(str(exc), title="Resume Refused (Model Config Changed)", border_style="red")
        )
        raise typer.Exit(code=1) from exc
    except LLMProviderError as exc:
        console.print(Panel(str(exc), title="LLM Provider Error", border_style="red"))
        raise typer.Exit(code=1) from exc
    except PipelineExecutionError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"[yellow]Partial report written to {out / 'report.md'}[/yellow]")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"Completed stages: {state.completed_stages}\n"
            f"Total elapsed: {state.total_elapsed_seconds:.2f}s\n"
            f"Definition: {out / 'definition.yaml'}\n"
            f"Report: {out / 'report.md'}\n"
            f"Scorecard CSV: {out / 'scorecard.csv'}",
            title="ForgeAI Pipeline Complete",
            title_align="left",
        )
    )


if __name__ == "__main__":
    app()
