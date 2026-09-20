"""Per-stage evaluation entry points and summary assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rag.chunking.semantic_chunker import ChunkingConfig, SemanticChunker
from rag.config import settings
from rag.embeddings.factory import build_embedding_service
from rag.evaluation.chunking_metrics import chunk_diagnostics, compare_configs
from rag.htmlx.section_extractor import SectionExtractor
from rag.logging_utils import get_logger
from rag.services.ingest_service import iter_documents

logger = get_logger(__name__)


def run_chunking_evaluation(
    file: Optional[Path] = None,
    directory: Optional[Path] = None,
    limit: int = 200,
    compare: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Stage 1: structural chunk quality, with optional A/B config comparison."""
    embeddings = build_embedding_service(provider=provider, model=model)
    extractor = SectionExtractor()
    documents = list(iter_documents(file=file, directory=directory, limit=limit))
    logger.info("Chunking evaluation over %d documents", len(documents))

    # Sections are independent of chunking config, so extract once.
    sections_by_doc = {d.document_id: extractor.extract(d.raw_html, d.retrieval_text)
                       for d in documents}
    section_count = sum(len(v) for v in sections_by_doc.values())

    configs = [ChunkingConfig.from_settings()]
    if compare:
        configs.append(
            ChunkingConfig(
                target_size=max(200, settings.CHUNK_TARGET_SIZE // 2),
                min_size=max(80, settings.CHUNK_MIN_SIZE // 2),
                max_size=max(400, settings.CHUNK_MAX_SIZE // 2),
                similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
                overlap_sentences=settings.CHUNK_OVERLAP,
            )
        )

    reports = []
    for config in configs:
        chunker = SemanticChunker(embeddings, config)
        chunks = []
        for document in documents:
            chunks.extend(chunker.chunk_document(document, sections_by_doc[document.document_id]))
        reports.append(
            chunk_diagnostics(
                chunks, len(documents), section_count, config.to_dict(),
                min_size=config.min_size, max_size=config.max_size,
            )
        )

    payload: dict = {
        "stage": "chunking",
        "documents": len(documents),
        "sections": section_count,
        "embedding_model": embeddings.model,
        "embedding_dimension": embeddings.dimension,
    }
    payload.update(compare_configs(reports) if compare else {"report": reports[0]})
    return payload


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_summary() -> tuple[dict, str]:
    """Merge whatever stage reports exist into one machine + human summary."""
    base = settings.EVALUATION_DIR
    chunking = _load(base / "chunking_results.json")
    retrieval = _load(base / "retrieval_results.json")
    rag = _load(base / "rag_results.json")

    summary = {
        "collection": settings.CHROMA_COLLECTION_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "chunking_config": {
            "target": settings.CHUNK_TARGET_SIZE,
            "min": settings.CHUNK_MIN_SIZE,
            "max": settings.CHUNK_MAX_SIZE,
            "similarity_threshold": settings.SEMANTIC_SIMILARITY_THRESHOLD,
            "overlap_sentences": settings.CHUNK_OVERLAP,
        },
        "stages": {
            "chunking": chunking or {"status": "not run"},
            "retrieval": (retrieval or {}).get("summary", {"status": "not run"}),
            "rag": (rag or {}).get("summary", {"status": "not run"}),
        },
    }
    return summary, _render(summary)


def _render(summary: dict) -> str:
    lines: list[str] = ["# RAG evaluation summary", ""]
    lines.append(f"- Collection: `{summary['collection']}`")
    lines.append(f"- Embedding model: `{summary['embedding_model']}`")
    lines.append(f"- LLM model: `{summary['llm_model']}`")
    lines.append(f"- Chunking: {summary['chunking_config']}")
    lines.append("")

    chunking = summary["stages"]["chunking"]
    report = chunking.get("report") if isinstance(chunking, dict) else None
    if report:
        size = report.get("size", {})
        lines += [
            "## Chunking",
            f"- Documents: {report.get('documents')}  Sections: {report.get('sections')}"
            f"  Chunks: {report.get('total_chunks')}",
            f"- Size: median={size.get('median')} p95={size.get('p95')} max={size.get('max')}",
            f"- Below min: {report.get('pct_below_min_size')}%  "
            f"At max: {report.get('pct_at_max_size')}%  "
            f"Duplicates: {report.get('duplicate_pct')}%",
            f"- Section boundary violations: {report.get('section_boundary_violations')}",
            "",
        ]

    retrieval = summary["stages"]["retrieval"]
    if isinstance(retrieval, dict) and retrieval.get("metrics"):
        metrics = retrieval["metrics"]
        lines += ["## Retrieval", f"- Questions evaluated: {retrieval.get('evaluated_questions')}"]
        for key in sorted(metrics):
            lines.append(f"- {key}: {metrics[key]}")
        lines.append(f"- Section retrieval accuracy: {retrieval.get('section_retrieval_accuracy')}")
        lines.append(f"- Source retrieval accuracy: {retrieval.get('source_retrieval_accuracy')}")
        latency = retrieval.get("latency", {}).get("total_retrieval", {})
        lines.append(
            f"- Latency ms: mean={latency.get('mean_ms')} median={latency.get('median_ms')} "
            f"p95={latency.get('p95_ms')} p99={latency.get('p99_ms')}"
        )
        lines.append("")

    rag = summary["stages"]["rag"]
    if isinstance(rag, dict) and (rag.get("deterministic_metrics") or rag.get("judge_metrics")):
        lines += ["## RAG answer quality", f"- Questions: {rag.get('evaluated_questions')}"]
        lines.append("### Deterministic")
        for key, value in sorted((rag.get("deterministic_metrics") or {}).items()):
            lines.append(f"- {key}: {value}")
        if rag.get("judge_metrics"):
            lines.append(f"### LLM-as-a-judge ({rag.get('judge_model')})")
            for key, value in sorted(rag["judge_metrics"].items()):
                lines.append(f"- {key}: {value}")
        if rag.get("unavailable_metrics"):
            lines.append("### Unavailable")
            for key, reason in sorted(rag["unavailable_metrics"].items()):
                lines.append(f"- {key}: {reason}")
        usage = rag.get("llm_usage") or {}
        lines.append(
            f"### LLM usage\n- input={usage.get('input_tokens')} "
            f"output={usage.get('output_tokens')} total={usage.get('total_tokens')} "
            f"cost_usd={usage.get('estimated_cost_usd')}"
        )
    return "\n".join(lines) + "\n"
