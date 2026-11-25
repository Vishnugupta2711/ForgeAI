"""Command-line interface for ForgeAI."""

from __future__ import annotations

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
from forge.config import load_config
from forge.ingest.profile import ingest_folder
from forge.llm.client import LLMClient
from forge.llm.models import LLMProviderError
from forge.retrieval.hybrid import HybridIndex, HybridRetriever
from forge.retrieval.models import RetrievalConfig
from forge.tools import ToolContext

app = typer.Typer(
    name="forge",
    help="ForgeAI: autonomous factory for building, testing, and refining agents.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """ForgeAI CLI root command."""


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
) -> None:
    """Ingest a document folder and print a compact corpus profile."""
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
) -> None:
    """Ingest/load a folder index and print ranked chunks with RRF scores."""
    cfg = load_config()
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
    llm_client: LLMClient | None = None,
) -> AgentRuntime:
    cfg = load_config().model_copy(deep=True)
    cfg.endpoints.provider = definition.model.provider
    cfg.models.agent = definition.model.name
    cfg.models.embed = definition.model.embed_model
    cfg.generation.temperature = definition.model.temperature
    cfg.generation.seed = definition.model.seed

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
) -> None:
    """Run an interactive multi-turn chat session using any declarative AgentDefinition."""
    definition = AgentDefinition.load_yaml(definition_path)
    runtime = _build_runtime_for_chat(
        definition, docs, run_dir=run_dir, bm25_only=bm25_only
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


if __name__ == "__main__":
    app()
