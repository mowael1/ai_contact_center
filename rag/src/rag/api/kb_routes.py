
"""Admin document-upload and agent-query endpoints for the per-company KB."""


from __future__ import annotations

import time
import httpx
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from rag.api.deps import get_kb_container, get_tenant
from rag.api.schemas import CitationModel, RetrievedChunkModel, UsageModel
from rag.config import settings
from rag.documents.loader import SUPPORTED
from rag.logging_utils import get_logger
from rag.services.conversation import (
    Conversation,
    Turn,
    contextual_query,
    conversations,
    format_history,
)
from rag.services.chat_store import chat_store
from rag.services.tenancy import Tenant
from rag.integrations.google_drive import upload_kb_file
from io import BytesIO
from fastapi.responses import StreamingResponse

from app.services.speech_service import (
    SpeechInputError,
    SpeechProviderError,
    synthesize_speech,
    transcribe_audio,
)

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
    drive_status: str = "disabled"
    drive_link: str = ""
    drive_error: str = ""


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
    
class STTResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str


class ChatSessionCreate(BaseModel):
    title: str = Field("New chat", min_length=1, max_length=120)


class ChatSessionModel(BaseModel):
    id: str
    company_id: int
    title: str
    summary: str = ""
    document_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ChatMessageModel(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime


class ChatMessagesResponse(BaseModel):
    session: ChatSessionModel
    messages: list[ChatMessageModel]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    persona: str = Field(
        "professional",
        min_length=1,
        max_length=100,
        description="Response style/persona requested by the frontend.",
    )
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


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class WebSearchResponse(BaseModel):
    query: str
    answer: str = ""
    sources: list[dict] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)


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
    "/web-search",
    response_model=WebSearchResponse,
    summary="Search the web through the configured n8n agent",
)
def web_search(
    payload: WebSearchRequest,
    tenant: Tenant = Depends(get_tenant),
) -> WebSearchResponse:
    """Delegate a web search to n8n without exposing the webhook to browsers."""
    webhook_url = settings.WEB_SEARCH_N8N_WEBHOOK_URL.strip()
    if not webhook_url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search is not configured. Set WEB_SEARCH_N8N_WEBHOOK_URL.",
        )

    try:
        response = httpx.post(
            webhook_url,
            json={"query": payload.query, "company_id": tenant.company_id},
            timeout=settings.WEB_SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError:
            # n8n may return the model output as text/plain when the
            # Respond to Webhook node is configured with a text response.
            result = {"answer": response.text}
    except httpx.ReadTimeout as exc:
        logger.exception("n8n web search timed out for company %s", tenant.company_id)
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "The web-search agent took too long to respond. "
                "Try again or increase WEB_SEARCH_TIMEOUT_SECONDS."
            ),
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("n8n web search failed for company %s", tenant.company_id)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="The web-search agent is currently unavailable.",
        ) from exc

    if isinstance(result, list):
        result = {"results": result}
    if not isinstance(result, dict):
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="The web-search agent returned an invalid response.",
        )

    raw_sources = result.get("sources", result.get("results", []))
    if not isinstance(raw_sources, list):
        raw_sources = []
    sources = [
        item if isinstance(item, dict) else {"title": str(item)}
        for item in raw_sources
    ]
    answer = result.get(
        "answer",
        result.get("response", result.get("text", result.get("output", ""))),
    )
    return WebSearchResponse(
        query=payload.query,
        answer=str(answer or ""),
        sources=sources,
        data=result,
    )


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
    drive_status = "disabled"
    drive_link = ""
    drive_error = ""
    try:
        drive_result = upload_kb_file(stored)
        if drive_result is not None:
            drive_status = "uploaded"
            drive_link = drive_result.get("web_view_link", "")
    except Exception:
        # Indexing remains useful if the optional Drive mirror is unavailable.
        logger.exception("Google Drive mirror upload failed")
        drive_status = "failed"
        drive_error = "Google Drive upload failed. Check OAuth/API setup and server logs."
    if result.status != "ingested":
        # Keep the file so an admin can inspect why it failed.
        return UploadResponse(
            status=result.status,
            company_id=tenant.company_id,
            collection=tenant.collection_name,
            source=result.source,
            pages=result.pages,
            quality=result.quality,
            reason=result.reason,
            drive_status=drive_status,
            drive_link=drive_link,
            drive_error=drive_error,
        )
    return UploadResponse(
        status="ingested",
        company_id=tenant.company_id,
        collection=tenant.collection_name,
        source=result.source,
        document_id=result.document_id,
        chunks=result.chunks,
        pages=result.pages,
        strategy=result.strategy,
        quality=result.quality,
        drive_status=drive_status,
        drive_link=drive_link,
        drive_error=drive_error,
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
    return {
        "deleted_chunks": removed,
        "document_id": document_id,
        "company_id": tenant.company_id,
    }


@router.post(
    "/sessions",
    response_model=ChatSessionModel,
    status_code=status.HTTP_201_CREATED,
)
def create_chat_session(
    payload: ChatSessionCreate = ChatSessionCreate(),
    tenant: Tenant = Depends(get_tenant),
) -> ChatSessionModel:
    return ChatSessionModel(**chat_store.create(tenant.company_id, payload.title))


@router.get("/sessions", response_model=list[ChatSessionModel])
def list_chat_sessions(tenant: Tenant = Depends(get_tenant)) -> list[ChatSessionModel]:
    return [ChatSessionModel(**item) for item in chat_store.list(tenant.company_id)]


@router.delete("/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    tenant: Tenant = Depends(get_tenant),
) -> dict:
    if not chat_store.delete(tenant.company_id, session_id):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    return {"deleted": True, "session_id": session_id}


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ChatMessagesResponse,
)
def get_chat_messages(
    session_id: str,
    tenant: Tenant = Depends(get_tenant),
) -> ChatMessagesResponse:
    item = chat_store.get(tenant.company_id, session_id)
    messages = chat_store.messages(tenant.company_id, session_id)
    if not item or messages is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    return ChatMessagesResponse(
        session=ChatSessionModel(**item),
        messages=[ChatMessageModel(**message) for message in messages],
    )

@router.post(
    "/stt",
    response_model=STTResponse,
    summary="Transcribe browser audio into text",
)
async def speech_to_text(
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_tenant),
) -> STTResponse:
    """Transcribe audio only.

    The returned text may subsequently be sent to /ask by the frontend.
    This endpoint never invokes the RAG pipeline itself.
    """

    try:
        audio_bytes = await file.read()

        text = transcribe_audio(
            audio_bytes,
            filename=file.filename or "recording.webm",
            content_type=file.content_type,
        )

        return STTResponse(
            text=text
        )

    except SpeechInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except SpeechProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech transcription is currently unavailable",
        ) from exc

    finally:
        await file.close()

@router.post(
    "/tts",
    summary="Generate Arabic WAV audio from text",
)
def text_to_speech(
    payload: TTSRequest,
    tenant: Tenant = Depends(get_tenant),
) -> StreamingResponse:
    """Generate audio only.

    This is a separate request from /ask, so TTS failures cannot remove
    or invalidate an existing text RAG answer.
    """

    try:
        audio_bytes = synthesize_speech(
            payload.text
        )

    except SpeechInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except SpeechProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech synthesis is currently unavailable",
        ) from exc

    return StreamingResponse(
        BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                'inline; filename="answer.wav"'
            ),
            "Cache-Control": "no-store",
        },
    )


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
    session_memory = (
        chat_store.memory(tenant.company_id, payload.conversation_id)
        if payload.conversation_id
        else None
    )

    if session_memory:
        recent = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in session_memory.recent
        )
        history = "\n".join(
            part
            for part in [
                (
                    f"Conversation summary:\n{session_memory.summary}"
                    if session_memory.summary
                    else ""
                ),
                f"Recent messages:\n{recent}" if recent else "",
            ]
            if part
        )

        # The durable transcript is the source of generation memory. The old
        # in-process store remains only for deterministic retrieval expansion.
        previous_user = next(
            (
                message["content"]
                for message in reversed(session_memory.recent)
                if message["role"] == "user"
            ),
            None,
        )

        conversation = Conversation(
            turns=[Turn(previous_user, "")] if previous_user else []
        )

        search_query = (
            contextual_query(payload.query, conversation)
            if conversation.turns
            else payload.query
        )
    else:
        conversation = conversations.get(
            tenant.company_id,
            payload.conversation_id,
        )
        search_query = contextual_query(payload.query, conversation)
        history = format_history(
            conversation,
            settings.CHAT_HISTORY_IN_PROMPT,
        )

    results = container.retriever.retrieve(
        search_query,
        top_k=payload.top_k,
    )

    answer = container.generator().generate(
        payload.query,
        results,
        history=history,
        persona=payload.persona,
    )

    answer.latency_ms = {
        **container.retriever.last_latency,
        **answer.latency_ms,
    }

    if payload.conversation_id:
        chat_store.append(
            tenant.company_id,
            payload.conversation_id,
            payload.query,
            answer.answer,
            [chunk.document_id for chunk in answer.retrieved],
        )

    return AskResponse(
        query=answer.query,
        search_query=search_query if search_query != payload.query else None,
        conversation_id=payload.conversation_id,
        answer=answer.answer,
        has_sufficient_context=answer.has_sufficient_context,
        citations=[
            CitationModel(**asdict(c), label=c.label())
            for c in answer.citations
        ],
        chunks=(
            [RetrievedChunkModel(**r.to_dict()) for r in answer.retrieved]
            if payload.include_chunks
            else None
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
    cleared = chat_store.delete(
        tenant.company_id,
        conversation_id,
    )
    conversations.clear(
        tenant.company_id,
        conversation_id,
    )
    return {
        "cleared": cleared,
        "conversation_id": conversation_id,
    }


@router.get(
    "/info",
    summary="Collection and model diagnostics",
)
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
