"""Admin document-upload and agent-query endpoints for the per-company KB."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from rag.api.deps import get_kb_container, get_tenant
from rag.api.schemas import CitationModel, RetrievedChunkModel, UsageModel
from rag.config import settings
from rag.documents.loader import SUPPORTED
from rag.logging_utils import get_logger
from rag.services.conversation import (
    contextual_query,
    conversations,
    format_history,
)
from rag.services.tenancy import Tenant

logger = get_logger(__name__)
router = APIRouter()


# ---- schemas --------------------------------------------------------------
class UploadResponse(BaseModel):
    status: str
    company_id: int
    collection: str
    source: str
    document_id: str = ""
    chunks: int = 0
    pages: int = 0
    strategy: str = ""
    quality: Optional[dict] = None
    reason: str = ""


class DocumentModel(BaseModel):
    document_id: str
    source: str
    document_title: str = ""
    source_type: str = ""
    chunks: int = 0


class DocumentListResponse(BaseModel):
    company_id: int
    collection: str
    vectors: int
    documents: list[DocumentModel]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    include_chunks: bool = False
    conversation_id: Optional[str] = Field(
        None, max_length=64,
        description=(
            "Opaque id that groups messages into a conversation, so follow-up "
            "questions resolve. Omit it for a one-off question."
        ),
    )


class AskResponse(BaseModel):
    query: str
    #: The query actually embedded. Differs from ``query`` when a follow-up was
    #: expanded with terms from the previous turn.
    search_query: Optional[str] = None
    conversation_id: Optional[str] = None
    answer: str
    has_sufficient_context: bool
    citations: list[CitationModel]
    chunks: Optional[list[RetrievedChunkModel]] = None
    usage: Optional[UsageModel] = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


# ---- helpers --------------------------------------------------------------
def _store_upload(tenant: Tenant, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED)}",
        )
    # Files are stored per company, so one tenant's uploads are never even on
    # the same path as another's.
    target_dir = Path(settings.KB_UPLOAD_DIR) / f"company_{tenant.company_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(upload.filename).name

    limit = settings.KB_MAX_UPLOAD_MB * 1024 * 1024
    with target.open("wb") as handle:
        written = 0
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {settings.KB_MAX_UPLOAD_MB} MB",
                )
            handle.write(chunk)
    return target


# ---- endpoints ------------------------------------------------------------
@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document into the caller's company knowledge base",
)
def upload_document(
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_tenant),
    container=Depends(get_kb_container),
) -> UploadResponse:
    """Store the file, extract, chunk and index it for this company only.

    Ingestion runs inline. It is fast for a help-centre PDF (a few seconds),
    but a large manual should move to a background worker before launch - see
    docs/multi-tenant-kb-plan.md.
    """
    stored = _store_upload(tenant, file)
    result = container.kb().ingest_file(stored, replace=True)
    if result.status != "ingested":
        # Keep the file so an admin can inspect why it failed.
        return UploadResponse(
            status=result.status, company_id=tenant.company_id,
            collection=tenant.collection_name, source=result.source,
            pages=result.pages, quality=result.quality, reason=result.reason,
        )
    return UploadResponse(
        status="ingested", company_id=tenant.company_id,
        collection=tenant.collection_name, source=result.source,
        document_id=result.document_id, chunks=result.chunks,
        pages=result.pages, strategy=result.strategy, quality=result.quality,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List the documents indexed for the caller's company",
)
def list_documents(
    tenant: Tenant = Depends(get_tenant),
    container=Depends(get_kb_container),
) -> DocumentListResponse:
    documents = container.store.list_documents()
    return DocumentListResponse(
        company_id=tenant.company_id,
        collection=tenant.collection_name,
        vectors=container.store.count(),
        documents=[DocumentModel(**d) for d in documents],
    )


@router.delete(
    "/documents/{document_id}",
    summary="Remove a document and its vectors",
)
def delete_document(
    document_id: str,
    tenant: Tenant = Depends(get_tenant),
    container=Depends(get_kb_container),
) -> dict:
    """Only ever deletes from the caller's own collection.

    A document id belonging to another company simply is not present here, so
    the call reports 0 removed rather than touching another tenant's data.
    """
    removed = container.kb().delete_document(document_id)
    if removed == 0:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="No such document in this company's knowledge base",
        )
    return {"deleted_chunks": removed, "document_id": document_id,
            "company_id": tenant.company_id}


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the agent assistant a question over this company's documents",
)
def ask(
    payload: AskRequest,
    tenant: Tenant = Depends(get_tenant),
    container=Depends(get_kb_container),
) -> AskResponse:
    """Retrieve from this company's collection, then answer with citations.

    With a ``conversation_id`` the previous turns are used twice: to expand a
    follow-up into a self-contained retrieval query, and as conversational
    context for the model. Facts still come only from retrieved passages.
    """
    conversation = conversations.get(tenant.company_id, payload.conversation_id)
    search_query = contextual_query(payload.query, conversation)
    history = format_history(conversation, settings.CHAT_HISTORY_IN_PROMPT)

    results = container.retriever.retrieve(search_query, top_k=payload.top_k)
    answer = container.generator().generate(payload.query, results, history=history)
    answer.latency_ms = {**container.retriever.last_latency, **answer.latency_ms}

    conversations.append(
        tenant.company_id, payload.conversation_id, payload.query, answer.answer
    )

    return AskResponse(
        query=answer.query,
        search_query=search_query if search_query != payload.query else None,
        conversation_id=payload.conversation_id,
        answer=answer.answer,
        has_sufficient_context=answer.has_sufficient_context,
        citations=[
            CitationModel(**asdict(c), label=c.label()) for c in answer.citations
        ],
        chunks=(
            [RetrievedChunkModel(**r.to_dict()) for r in answer.retrieved]
            if payload.include_chunks else None
        ),
        usage=UsageModel(**answer.usage.to_dict()) if answer.usage else None,
        latency_ms=answer.latency_ms,
    )


@router.delete(
    "/conversations/{conversation_id}",
    summary="Forget a conversation's history",
)
def clear_conversation(
    conversation_id: str,
    tenant: Tenant = Depends(get_tenant),
) -> dict:
    """Start fresh. Scoped to the caller's company, so ids cannot collide."""
    cleared = conversations.clear(tenant.company_id, conversation_id)
    return {"cleared": cleared, "conversation_id": conversation_id}


@router.get("/info", summary="Collection and model diagnostics")
def info(
    tenant: Tenant = Depends(get_tenant),
    container=Depends(get_kb_container),
) -> dict:
    return {
        "company_id": tenant.company_id,
        # True when a super admin is viewing a company they do not belong to.
        # Surfaced so the UI can make the borrowed scope obvious.
        "delegated": tenant.is_delegated,
        "collection": tenant.collection_name,
        "vectors": container.store.count(),
        "embedding": container.embeddings.model_info(),
        "chunking": {
            "size": settings.CHUNK_SIZE_TOKENS,
            "overlap": settings.CHUNK_OVERLAP_TOKENS,
            "unit": settings.CHUNK_UNIT,
        },
        "memory": {
            "turns_kept": settings.CHAT_MEMORY_TURNS,
            "turns_in_prompt": settings.CHAT_HISTORY_IN_PROMPT,
            **conversations.stats(),
        },
        "auth_mode": settings.RAG_AUTH_MODE,
    }
