"""Command-line interface for the whole RAG backend.

Built on Typer + Rich, matching the parent project's dependency set. Every
stage of the pipeline is inspectable from here:

    rag-cli dataset inspect
    rag-cli ingest run
    rag-cli inspect sections --url ...
    rag-cli inspect chunks   --url ...
    rag-cli retrieve  --query ...
    rag-cli ask       --query ...
    rag-cli store info
    rag-cli eval retrieval | rag | chunking
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.config import settings
from rag.logging_utils import configure_logging

app = typer.Typer(help="Vodafone Egypt RAG backend", no_args_is_help=True, add_completion=False)
dataset_app = typer.Typer(help="Dataset inspection and diagnostics", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingestion into Chroma Cloud", no_args_is_help=True)
inspect_app = typer.Typer(help="Inspect sections and chunks", no_args_is_help=True)
store_app = typer.Typer(help="Vector store information", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(ingest_app, name="ingest")
app.add_typer(inspect_app, name="inspect")
app.add_typer(store_app, name="store")
app.add_typer(eval_app, name="eval")

console = Console()

# ---- shared options ------------------------------------------------------
FileOpt = typer.Option(None, "--file", "-f", help="A single .parquet file")
DirOpt = typer.Option(None, "--directory", "-d", help="Directory of .parquet files")
LimitOpt = typer.Option(None, "--limit", "-n", help="Stop after N rows")
OfflineOpt = typer.Option(
    False, "--offline",
    help="Use the in-memory vector store instead of Chroma Cloud (no credentials needed)",
)
OfflineStoreOpt = typer.Option(
    None, "--offline-store", help="JSON snapshot path for the offline store"
)
EmbedProviderOpt = typer.Option(None, "--embedding-provider", help="Override EMBEDDING_PROVIDER")
EmbedModelOpt = typer.Option(None, "--embedding-model", help="Override EMBEDDING_MODEL")


@app.callback()
def _main(log_level: str = typer.Option(settings.LOG_LEVEL, "--log-level")) -> None:
    configure_logging(log_level)


def _container(offline, offline_store, provider=None, model=None):
    from rag.container import build_container

    return build_container(
        offline=offline,
        offline_path=Path(offline_store) if offline_store else None,
        embedding_provider=provider,
        embedding_model=model,
    )


def _print_json(payload: dict) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False, default=str))


# ==========================================================================
# dataset
# ==========================================================================
@dataset_app.command("inspect")
def dataset_inspect(
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: Optional[int] = LimitOpt,
    samples: int = typer.Option(0, "--samples", help="Print N sample records"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write the report to a file"),
) -> None:
    """Field coverage, status breakdown, duplicates, language mix, lengths."""
    from rag.evaluation.diagnostics import dataset_report

    report = dataset_report(file=file, directory=directory, limit=limit, samples=samples)
    _print_json(report)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {json_out}")


@dataset_app.command("extraction")
def dataset_extraction(
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: int = typer.Option(200, "--limit", "-n"),
    json_out: Optional[Path] = typer.Option(None, "--json"),
) -> None:
    """HTML section-extraction coverage: methods, fallbacks, heading spread."""
    from rag.evaluation.diagnostics import extraction_report

    report = extraction_report(file=file, directory=directory, limit=limit)
    _print_json(report)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


# ==========================================================================
# ingest
# ==========================================================================
@ingest_app.command("run")
def ingest_run(
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: Optional[int] = LimitOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="Chunk but do not write vectors"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
) -> None:
    """Parquet -> sections -> chunks -> embeddings -> vector store (idempotent)."""
    container = _container(offline, offline_store, provider, model)
    pipeline = container.ingestion()
    stats = pipeline.run(file=file, directory=directory, limit=limit, dry_run=dry_run)
    _print_json(stats.to_dict())
    if offline and offline_store:
        container.store.save(Path(offline_store))  # type: ignore[attr-defined]
        console.print(f"[green]Saved offline store[/green] {offline_store}")


@ingest_app.command("delete")
def ingest_delete(
    document_id: str = typer.Option(..., "--document-id"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
) -> None:
    """Remove every chunk of a document, so it can be cleanly re-ingested."""
    container = _container(offline, offline_store)
    removed = container.store.delete_document(document_id)
    console.print(f"Deleted [bold]{removed}[/bold] chunks for {document_id}")


# ==========================================================================
# inspect
# ==========================================================================
def _find_document(url: Optional[str], document_id: Optional[str], file, directory, limit):
    from rag.ingestion.document_builder import document_id_for
    from rag.services.ingest_service import iter_documents

    target_id = document_id or (document_id_for(url) if url else None)
    for document in iter_documents(file=file, directory=directory, limit=limit):
        if target_id is None:
            return document
        if document.document_id == target_id:
            return document
    return None


@inspect_app.command("sections")
def inspect_sections(
    url: Optional[str] = typer.Option(None, "--url"),
    document_id: Optional[str] = typer.Option(None, "--document-id"),
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: Optional[int] = typer.Option(3000, "--limit", "-n"),
    show_text: int = typer.Option(200, "--show-text", help="Characters of sample content"),
) -> None:
    """Validate the HTML parser: show every section, its path and its method."""
    from rag.htmlx.section_extractor import SectionExtractor

    document = _find_document(url, document_id, file, directory, limit)
    if document is None:
        console.print("[red]Document not found.[/red]")
        raise typer.Exit(1)

    sections = SectionExtractor().extract(document.raw_html, document.retrieval_text)
    console.print(Panel.fit(
        f"[bold]Document:[/bold] {document.document_id}\n"
        f"[bold]URL:[/bold] {document.source_url}\n"
        f"[bold]Language:[/bold] {document.language}   "
        f"[bold]HTML:[/bold] {len(document.raw_html or ''):,} chars\n"
        f"[bold]Sections:[/bold] {len(sections)}",
        title="Section inspection",
    ))
    if not sections:
        console.print("[yellow]No sections extracted (fallback status: no_section).[/yellow]")
        return

    methods: dict[str, int] = {}
    levels: dict[str, int] = {}
    for section in sections:
        methods[section.extraction_method] = methods.get(section.extraction_method, 0) + 1
        key = str(section.heading_level)
        levels[key] = levels.get(key, 0) + 1

    for section in sections:
        label = section.extraction_method
        if section.component_type:
            label += f"/{section.component_type}"
        console.print(
            f"\n[bold cyan]Section {section.section_index + 1}:[/bold cyan] "
            f"{section.path_string}"
        )
        console.print(
            f"   level=[bold]{section.heading_level}[/bold]  method=[bold]{label}[/bold]  "
            f"chars={len(section.text)}"
        )
        if show_text:
            console.print(f"   [dim]{section.text[:show_text]}[/dim]")

    table = Table(title="Summary", show_header=True)
    table.add_column("Extraction method"); table.add_column("Sections", justify="right")
    for method, count in sorted(methods.items()):
        table.add_row(method, str(count))
    console.print(table)
    console.print(f"Heading levels: {levels}")
    console.print(
        "Fallback status: "
        + ("[yellow]text_fallback / no_section in use[/yellow]"
           if methods.keys() & {"text_fallback", "no_section"}
           else "[green]none - HTML structure sufficed[/green]")
    )


@inspect_app.command("chunks")
def inspect_chunks(
    url: Optional[str] = typer.Option(None, "--url"),
    document_id: Optional[str] = typer.Option(None, "--document-id"),
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: Optional[int] = typer.Option(3000, "--limit", "-n"),
    show_text: int = typer.Option(240, "--show-text"),
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
) -> None:
    """Show sentence counts, chunk sizes, boundaries and sample chunk text."""
    from rag.chunking.semantic_chunker import SemanticChunker
    from rag.chunking.sentence_segmenter import split_sentences
    from rag.embeddings.factory import build_embedding_service
    from rag.htmlx.section_extractor import SectionExtractor

    document = _find_document(url, document_id, file, directory, limit)
    if document is None:
        console.print("[red]Document not found.[/red]")
        raise typer.Exit(1)

    embeddings = build_embedding_service(provider=provider, model=model)
    sections = SectionExtractor().extract(document.raw_html, document.retrieval_text)
    chunks = SemanticChunker(embeddings).chunk_document(document, sections)

    console.print(Panel.fit(
        f"[bold]Document:[/bold] {document.document_id}\n"
        f"[bold]URL:[/bold] {document.source_url}\n"
        f"[bold]Sections:[/bold] {len(sections)}   [bold]Chunks:[/bold] {len(chunks)}\n"
        f"[bold]Embedding:[/bold] {embeddings.model}",
        title="Chunk inspection",
    ))

    by_section: dict[int, list] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section_index, []).append(chunk)

    for section in sections:
        section_chunks = by_section.get(section.section_index, [])
        sentences = split_sentences(section.text)
        console.print(
            f"\n[bold cyan]Section:[/bold cyan] {section.path_string}"
            f"\n   method=[bold]{section.extraction_method}[/bold]"
            f"  sentences=[bold]{len(sentences)}[/bold]"
            f"  chars={len(section.text)}"
            f"  chunks=[bold]{len(section_chunks)}[/bold]"
        )
        for chunk in section_chunks:
            console.print(
                f"   [green]Chunk {chunk.chunk_index + 1}[/green] "
                f"({chunk.char_len()} chars, {chunk.sentence_count} sentences, "
                f"{chunk.chunking_method})  id={chunk.chunk_id}"
            )
            if show_text:
                console.print(f"      [dim]{chunk.text[:show_text]}[/dim]")

    if chunks:
        sizes = [c.char_len() for c in chunks]
        console.print(
            f"\nChunk sizes: min={min(sizes)} max={max(sizes)} "
            f"mean={sum(sizes) // len(sizes)}  "
            f"(limits {settings.CHUNK_MIN_SIZE}/{settings.CHUNK_TARGET_SIZE}/{settings.CHUNK_MAX_SIZE})"
        )


# ==========================================================================
# retrieve / ask
# ==========================================================================
@app.command("retrieve")
def retrieve_cmd(
    query: str = typer.Option(..., "--query", "-q"),
    top_k: int = typer.Option(settings.TOP_K, "--top-k", "-k"),
    domain: Optional[str] = typer.Option(None, "--domain"),
    language: Optional[str] = typer.Option(None, "--language"),
    document_id: Optional[str] = typer.Option(None, "--document-id"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieval only - no generation."""
    container = _container(offline, offline_store, provider, model)
    filters = {"domain": domain, "language": language, "document_id": document_id}
    results = container.retriever.retrieve(query, top_k=top_k, filters=filters)

    if json_out:
        _print_json({
            "query": query,
            "results": [r.to_dict() for r in results],
            "latency_ms": container.retriever.last_latency,
        })
        return

    console.print(Panel.fit(f"[bold]{query}[/bold]", title="Query"))
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return
    for i, result in enumerate(results, 1):
        console.print(
            f"\n[bold green]{i}.[/bold green] score=[bold]{result.score:.4f}[/bold]  "
            f"{result.source_url}"
        )
        console.print(f"   section: {' > '.join(result.section_path or []) or '(none)'}")
        console.print(f"   [dim]{result.text[:300]}[/dim]")
    console.print(f"\n[dim]latency: {container.retriever.last_latency}[/dim]")


@app.command("ask")
def ask_cmd(
    query: str = typer.Option(..., "--query", "-q"),
    top_k: int = typer.Option(settings.TOP_K, "--top-k", "-k"),
    stream: bool = typer.Option(
        settings.LLM_STREAMING, "--stream/--no-stream",
        help="Print the answer token by token as it is generated",
    ),
    agentic: Optional[bool] = typer.Option(
        None, "--agentic/--linear",
        help="LangGraph retrieve/evaluate/rewrite loop, or the linear pipeline",
    ),
    max_attempts: Optional[int] = typer.Option(
        None, "--max-attempts", help="Retrieval attempts including the first"
    ),
    show_trace: bool = typer.Option(
        False, "--trace", help="Print the agentic loop's per-attempt audit trail"
    ),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Full RAG: retrieve, evaluate, generate a grounded answer, cite."""
    container = _container(offline, offline_store, provider, model)
    answerer = container.answerer(agentic=agentic, max_attempts=max_attempts)

    console.print(Panel.fit(f"[bold]{query}[/bold]", title="Question"))

    if stream and not json_out:
        # Print deltas as they arrive; the final event carries the real answer.
        console.print("[bold]Answer:[/bold] ", end="")
        printed = 0
        answer = None
        for piece in answerer.query_stream(query, top_k=top_k):
            if piece.done:
                answer = piece.answer
                break
            if piece.replaces_from is not None and piece.replaces_from < printed:
                # Post-processing rewrote the tail: reprint the corrected answer.
                console.print()
                console.print(f"[dim](revised)[/dim] {piece.text}")
                printed = len(piece.text)
                continue
            print(piece.delta, end="", flush=True)
            printed += len(piece.delta)
        print()
    else:
        answer = answerer.query(query, top_k=top_k)

    if answer is None:
        console.print("[red]No answer produced.[/red]")
        raise typer.Exit(1)

    if json_out:
        _print_json({
            "query": answer.query,
            "answer": answer.answer,
            "has_sufficient_context": answer.has_sufficient_context,
            "citations": [asdict(c) for c in answer.citations],
            "attempts": answer.attempts,
            "trace": answer.trace,
            "usage": answer.usage.to_dict() if answer.usage else None,
            "latency_ms": answer.latency_ms,
        })
        return

    if not stream:
        console.print(Panel(answer.answer, title="Answer"))

    if show_trace and answer.attempts:
        table = Table(title="Agentic retrieval loop")
        table.add_column("#"); table.add_column("Query"); table.add_column("Hits", justify="right")
        table.add_column("Relevant"); table.add_column("Conf.", justify="right")
        table.add_column("Reason")
        for a in answer.attempts:
            table.add_row(
                str(a.get("attempt")), (a.get("query") or "")[:45],
                str(a.get("retrieved")),
                "[green]yes[/green]" if a.get("is_relevant") else "[yellow]no[/yellow]",
                f"{a.get('confidence', 0):.2f}", (a.get("reason") or "")[:50],
            )
        console.print(table)

    if answer.citations:
        table = Table(title="Citations")
        table.add_column("#"); table.add_column("Section"); table.add_column("Source URL")
        for citation in answer.citations:
            table.add_row(
                str(citation.index),
                " > ".join(citation.section_path or []) or (citation.section_title or "-"),
                citation.source_url,
            )
        console.print(table)
    else:
        console.print("[yellow]No citations (insufficient context).[/yellow]")
    if answer.usage:
        console.print(f"[dim]usage: {answer.usage.to_dict()}[/dim]")
    console.print(f"[dim]latency: {answer.latency_ms}[/dim]")


# ==========================================================================
# store
# ==========================================================================
@store_app.command("info")
def store_info(
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
) -> None:
    """Collection name, vector count, embedding model, connection status.

    The API key is never printed - only whether one is set.
    """
    container = _container(offline, offline_store, provider, model)
    info = container.store.info()
    info["embedding"] = container.embeddings.model_info()
    _print_json(info)


@store_app.command("config")
def store_config() -> None:
    """Print effective configuration with every secret redacted."""
    _print_json(settings.safe_dump())


# ==========================================================================
# eval
# ==========================================================================
@eval_app.command("build-dataset")
def eval_build_dataset(
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: int = typer.Option(600, "--limit", "-n"),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Derive a curated evaluation set from real dataset content."""
    from rag.evaluation.dataset_builder import build_evaluation_dataset

    target = out or settings.EVALUATION_DIR / "retrieval_eval.json"
    dataset = build_evaluation_dataset(file=file, directory=directory, limit=limit)
    dataset.save(target)
    console.print(f"[green]Wrote[/green] {target} with {len(dataset.questions)} questions")


@eval_app.command("chunking")
def eval_chunking(
    file: Optional[Path] = FileOpt,
    directory: Optional[Path] = DirOpt,
    limit: int = typer.Option(200, "--limit", "-n"),
    compare: bool = typer.Option(False, "--compare", help="Compare two chunking configs"),
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Stage 1: chunk-quality diagnostics, independent of retrieval."""
    from rag.evaluation.stages import run_chunking_evaluation

    report = run_chunking_evaluation(
        file=file, directory=directory, limit=limit, compare=compare,
        provider=provider, model=model,
    )
    _print_json(report)
    target = out or settings.EVALUATION_DIR / "chunking_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {target}")


@eval_app.command("retrieval")
def eval_retrieval(
    dataset_path: Optional[Path] = typer.Option(None, "--dataset"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Stage 2: HitRate/Recall/Precision/MRR/nDCG + latency."""
    from rag.evaluation.dataset import EvalDataset
    from rag.evaluation.runner import evaluate_retrieval, write_report

    container = _container(offline, offline_store, provider, model)
    dataset = EvalDataset.load(dataset_path)
    if limit:
        dataset = dataset.sample(limit)
    result = evaluate_retrieval(container.retriever, dataset, top_k=top_k)
    _print_json(result.summary)
    write_report(
        {"summary": result.summary, "per_query": result.per_query},
        out or settings.EVALUATION_DIR / "retrieval_results.json",
    )


@eval_app.command("rag")
def eval_rag(
    dataset_path: Optional[Path] = typer.Option(None, "--dataset"),
    top_k: int = typer.Option(settings.TOP_K, "--top-k", "-k"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Evaluate a balanced subset - useful under a provider daily quota",
    ),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Enable LLM-as-a-judge metrics"),
    agentic: Optional[bool] = typer.Option(
        None, "--agentic/--linear",
        help="Evaluate the LangGraph loop, or the linear pipeline",
    ),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Stage 3-6: answer, context and citation quality end to end."""
    from rag.evaluation.dataset import EvalDataset
    from rag.evaluation.rag_metrics import LLMJudge
    from rag.evaluation.runner import evaluate_rag, write_report

    container = _container(offline, offline_store, provider, model)
    dataset = EvalDataset.load(dataset_path)
    if limit:
        dataset = dataset.sample(limit)
        console.print(f"[yellow]Evaluating a balanced subset of {len(dataset.questions)} questions[/yellow]")
    judge_impl = LLMJudge(container.judge_llm) if judge else None
    answerer = container.answerer(agentic=agentic)
    result = evaluate_rag(answerer, dataset, top_k=top_k, judge=judge_impl)
    _print_json(result.summary)
    write_report(
        {"summary": result.summary, "per_query": result.per_query},
        out or settings.EVALUATION_DIR / "rag_results.json",
    )


@eval_app.command("summary")
def eval_summary(
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Merge the stage reports into evaluation_summary.json + a readable digest."""
    from rag.evaluation.stages import build_summary

    summary, text = build_summary()
    target = out or settings.EVALUATION_DIR / "evaluation_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (target.parent / "evaluation_summary.md").write_text(text, encoding="utf-8")
    console.print(text)
    console.print(f"[green]Wrote[/green] {target}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
