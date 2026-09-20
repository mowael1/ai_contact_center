"""RAG API routes: /health, /retrieve, /query."""

from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from rag.api.schemas import (
    AttemptModel,
    CitationModel,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkModel,
    RetrieveRequest,
    RetrieveResponse,
    UsageModel,
)
from rag.container import Container, get_container
from rag.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter()
health_router = APIRouter()


def _to_chunk_models(chunks) -> list[RetrievedChunkModel]:
    return [RetrievedChunkModel(**c.to_dict()) for c in chunks]


@health_router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(container: Container = Depends(get_container)) -> HealthResponse:
    """Liveness plus vector-store / embedding diagnostics. Never returns secrets."""
    try:
        store_info = container.store.info()
        healthy = bool(store_info.get("connected", False))
    except Exception as exc:
        store_info = {"connected": False, "error": type(exc).__name__}
        healthy = False
    try:
        embedding_info = container.embeddings.model_info()
    except Exception as exc:
        embedding_info = {"error": type(exc).__name__}
    return HealthResponse(
        status="ok" if healthy else "degraded",
        vector_store=store_info,
        embedding=embedding_info,
    )


@router.post("/retrieve", response_model=RetrieveResponse, summary="Retrieve chunks only")
def retrieve(
    payload: RetrieveRequest, container: Container = Depends(get_container)
) -> RetrieveResponse:
    """Vector search with no generation. Use this to inspect grounding."""
    filters = payload.filters.as_dict() if payload.filters else None
    try:
        results = container.retriever.retrieve(
            payload.query, top_k=payload.top_k, filters=filters
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval failed: {type(exc).__name__}",
        ) from exc
    return RetrieveResponse(
        query=payload.query,
        top_k=payload.top_k,
        count=len(results),
        results=_to_chunk_models(results),
        latency_ms=container.retriever.last_latency,
    )


@router.post("/query", response_model=QueryResponse, summary="Full grounded RAG answer")
def query(
    payload: QueryRequest, container: Container = Depends(get_container)
) -> QueryResponse:
    """Retrieve, build context, generate a grounded answer, attach citations."""
    filters = payload.filters.as_dict() if payload.filters else None
    try:
        answerer = container.answerer(
            agentic=payload.agentic, max_attempts=payload.max_attempts
        )
        answer = answerer.query(payload.query, top_k=payload.top_k, filters=filters)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG query failed: {type(exc).__name__}",
        ) from exc

    return _to_query_response(answer, include_chunks=payload.include_chunks)


def _to_query_response(answer, include_chunks: bool) -> QueryResponse:
    return QueryResponse(
        query=answer.query,
        answer=answer.answer,
        has_sufficient_context=answer.has_sufficient_context,
        citations=[CitationModel(**asdict(c)) for c in answer.citations],
        chunks=_to_chunk_models(answer.retrieved) if include_chunks else None,
        usage=UsageModel(**answer.usage.to_dict()) if answer.usage else None,
        latency_ms=answer.latency_ms,
        attempts=[AttemptModel(**a) for a in (answer.attempts or [])],
        trace=list(answer.trace or []),
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/query/stream",
    summary="Full grounded RAG answer, streamed as Server-Sent Events",
    response_class=StreamingResponse,
)
def query_stream(
    payload: QueryRequest, container: Container = Depends(get_container)
) -> StreamingResponse:
    """Stream the answer token by token over SSE.

    Events:

    * ``delta`` - ``{"delta": "...", "replaces_from": null}``. Append ``delta``;
      if ``replaces_from`` is an integer, truncate to that offset first (the
      final reconciliation event uses it after citation post-processing).
    * ``done``  - the complete :class:`QueryResponse` payload, including
      citations, usage and the agentic audit trail.
    * ``error`` - ``{"detail": "..."}`` if generation fails mid-stream.

    Citations are only ever sent on ``done``: they are resolved from the
    finished answer against retrieved metadata, so they cannot be invented
    mid-stream.
    """
    filters = payload.filters.as_dict() if payload.filters else None
    answerer = container.answerer(
        agentic=payload.agentic, max_attempts=payload.max_attempts
    )

    def events():
        try:
            for piece in answerer.query_stream(
                payload.query, top_k=payload.top_k, filters=filters
            ):
                if piece.done and piece.answer is not None:
                    response = _to_query_response(
                        piece.answer, include_chunks=payload.include_chunks
                    )
                    yield _sse("done", response.model_dump(mode="json"))
                elif piece.delta:
                    yield _sse(
                        "delta",
                        {"delta": piece.delta, "replaces_from": piece.replaces_from},
                    )
        except Exception as exc:  # the stream has already started; report in-band
            logger.exception("Streaming RAG query failed")
            yield _sse("error", {"detail": f"generation failed: {type(exc).__name__}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
