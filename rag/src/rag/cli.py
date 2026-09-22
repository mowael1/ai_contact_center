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
store_app = typer.Typer(help="Vector store information", no_args_is_help=True)
kb_app = typer.Typer(help="Per-company knowledge base (uploaded files)", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation", no_args_is_help=True)
app.add_typer(kb_app, name="kb")
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
# kb  (per-company uploaded documents)
# ==========================================================================
CompanyOpt = typer.Option(..., "--company-id", "-c", help="Tenant company id")


def _tenant_container(company_id, offline, offline_store, provider=None, model=None):
    from rag.container import build_tenant_container

    return build_tenant_container(
        company_id=company_id, offline=offline,
        offline_path=Path(offline_store) if offline_store else None,
        embedding_provider=provider, embedding_model=model,
    )


@kb_app.command("inspect")
def kb_inspect(
    file: Path = typer.Option(..., "--file", "-f", help="A document to inspect"),
    show_text: int = typer.Option(160, "--show-text"),
) -> None:
    """Load, quality-check, section and chunk a file WITHOUT indexing it.

    Use this first on real customer documents: if extraction is damaged,
    everything downstream inherits the damage.
    """
    from rag.chunking.strategies import ChunkingSettings, chunk_text
    from rag.documents.loader import load_document
    from rag.documents.quality import assess, repair_extracted
    from rag.documents.sections import build_sections

    document = load_document(file)
    repairs: list[str] = []
    for page in document.pages:
        page.text, info = repair_extracted(page.text)
        if info["reversal"]["is_reversed"]:
            repairs.append(f"page {page.page_number}: reversed Arabic repaired")
    if repairs:
        for heading in document.headings:
            heading.text = heading.text[::-1]
        document.title = document.title[::-1]
    quality = assess(document)

    console.print(Panel.fit(
        f"[bold]{document.title}[/bold]\n"
        f"source: {document.source}   type: {document.source_type}\n"
        f"pages: {document.page_count}   chars: {len(document.full_text):,}   "
        f"headings: {len(document.headings)}",
        title="Document",
    ))

    status = "[green]OK[/green]" if quality.ok else "[red]BLOCKED[/red]"
    console.print(f"\nExtraction quality: {status}"
                  + ("  [yellow](needs OCR)[/yellow]" if quality.needs_ocr else ""))
    table = Table(show_header=False)
    for key, value in quality.to_dict().items():
        if key != "problems":
            table.add_row(key, str(value))
    console.print(table)
    for problem in quality.problems:
        console.print(f"  [yellow]![/yellow] {problem}")
    for repaired in repairs:
        console.print(f"  [green]fixed[/green] {repaired}")

    sections = build_sections(document)
    config = ChunkingSettings.from_settings()
    parts, strategy = chunk_text(document.full_text, sections, config)

    console.print(f"\n[bold]Strategy chosen:[/bold] [cyan]{strategy}[/cyan]  "
                  f"(config: {config.to_dict()})")
    console.print(f"[bold]Sections:[/bold] {len(sections)}   [bold]Chunks:[/bold] {len(parts)}\n")
    for part in parts:
        head = part.section_path and " > ".join(part.section_path) or "(no section)"
        console.print(f"[green]part {part.part}[/green] [{part.strategy}] "
                      f"{part.unit_count} {config.unit}, {len(part.text)} chars"
                      f"  page {part.page_number or '-'}")
        console.print(f"   [dim]{head}[/dim]")
        if show_text:
            console.print(f"   {part.text[:show_text]}\n")


@kb_app.command("ingest")
def kb_ingest(
    company_id: int = CompanyOpt,
    path: Path = typer.Option(..., "--path", "-p", help="File or directory to ingest"),
    replace: bool = typer.Option(True, "--replace/--no-replace",
                                 help="Replace a document's existing chunks"),
    export: Optional[Path] = typer.Option(None, "--export",
                                          help="Also write chunks.json here"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
) -> None:
    """Ingest a company's documents into its own collection."""
    container = _tenant_container(company_id, offline, offline_store, provider, model)
    kb = container.kb()
    report = kb.ingest_path(path, replace=replace)
    _print_json(report.to_dict())

    if export:
        chunks: list = []
        for result in report.results:
            if result.status != "ingested":
                continue
            produced, _, _, _ = kb.chunk_file(result.path)
            chunks.extend(produced)
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(
            json.dumps([c.to_export() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/green] {export} ({len(chunks)} chunks)")

    if offline and offline_store:
        container.store.save(Path(offline_store))  # type: ignore[attr-defined]


@kb_app.command("list")
def kb_list(
    company_id: int = CompanyOpt,
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
) -> None:
    """List the documents indexed for one company."""
    container = _tenant_container(company_id, offline, offline_store, provider)
    documents = container.store.list_documents()
    console.print(f"collection: [bold]{container.tenant.collection_name}[/bold]   "
                  f"vectors: {container.store.count()}")
    if not documents:
        console.print("[yellow]No documents indexed.[/yellow]")
        return
    table = Table()
    table.add_column("document_id"); table.add_column("source")
    table.add_column("title"); table.add_column("chunks", justify="right")
    for doc in documents:
        table.add_row(doc["document_id"], doc["source"],
                      doc.get("document_title", ""), str(doc["chunks"]))
    console.print(table)


@kb_app.command("delete")
def kb_delete(
    company_id: int = CompanyOpt,
    document_id: str = typer.Option(..., "--document-id"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
) -> None:
    """Delete one document's chunks from a company's collection."""
    container = _tenant_container(company_id, offline, offline_store, provider)
    removed = container.kb().delete_document(document_id)
    console.print(f"Deleted [bold]{removed}[/bold] chunks for {document_id}")


@kb_app.command("ask")
def kb_ask(
    company_id: int = CompanyOpt,
    query: str = typer.Option(..., "--query", "-q"),
    top_k: int = typer.Option(settings.TOP_K, "--top-k", "-k"),
    retrieve_only: bool = typer.Option(False, "--retrieve-only"),
    offline: bool = OfflineOpt,
    offline_store: Optional[str] = OfflineStoreOpt,
    provider: Optional[str] = EmbedProviderOpt,
    model: Optional[str] = EmbedModelOpt,
) -> None:
    """Ask a question against ONE company's knowledge base."""
    container = _tenant_container(company_id, offline, offline_store, provider, model)
    console.print(Panel.fit(
        f"[bold]{query}[/bold]\ncompany {company_id} -> {container.tenant.collection_name}",
        title="Question"))

    results = container.retriever.retrieve(query, top_k=top_k)
    if retrieve_only or not results:
        for i, r in enumerate(results, 1):
            meta = r.metadata
            console.print(f"\n[bold green]{i}.[/bold green] score={r.score:.4f}  "
                          f"{meta.get('source')} (page {meta.get('page_number')})")
            console.print(f"   section: {' > '.join(r.section_path or []) or '-'}")
            console.print(f"   [dim]{r.text[:260]}[/dim]")
        if not results:
            console.print("[yellow]No results in this company's knowledge base.[/yellow]")
        return

    answer = container.generator().generate(query, results)
    console.print(Panel(answer.answer, title="Answer"))
    if answer.citations:
        table = Table(title="Citations")
        table.add_column("#"); table.add_column("Document"); table.add_column("Section")
        table.add_column("Page", justify="right")
        by_id = {c.chunk_id: c for c in results}
        for citation in answer.citations:
            chunk = by_id.get(citation.chunk_id)
            meta = chunk.metadata if chunk else {}
            table.add_row(str(citation.index), meta.get("source", "-"),
                          " > ".join(citation.section_path or []) or "-",
                          str(meta.get("page_number", "-")))
        console.print(table)
    if answer.usage:
        console.print(f"[dim]usage: {answer.usage.to_dict()}[/dim]")


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
