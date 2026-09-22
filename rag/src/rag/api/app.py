"""Standalone FastAPI application for the RAG backend.

Also exposes :data:`rag_router`, which the parent ``app/`` project can mount
under its own ``/api/v1`` without importing anything else from this package.

There is no UI here by design - backend services, REST and CLI only.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from rag.config import RAG_ROOT

from rag.api.kb_routes import router as kb_router
from rag.api.routes import health_router, router
from rag.logging_utils import configure_logging

#: Mount this in any FastAPI app: ``app.include_router(rag_router, prefix="/api/v1")``
rag_router = APIRouter()
rag_router.include_router(router, prefix="/rag", tags=["RAG"])
rag_router.include_router(kb_router, prefix="/kb", tags=["Knowledge Base"])


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title="Vodafone Egypt RAG API",
        version="1.0.0",
        description=(
            "Arabic/English retrieval-augmented generation over the Vodafone "
            "Egypt web knowledge base. HTML-first section extraction, semantic "
            "chunking, Chroma Cloud retrieval and grounded generation with "
            "citations built from retrieved metadata."
        ),
    )
    application.include_router(health_router)
    application.include_router(rag_router, prefix="/api/v1")

    # The admin upload page and agent console are served from the repo's
    # existing frontend, so the whole flow can be exercised in a browser.
    frontend = RAG_ROOT.parent / "frontend"
    if frontend.exists():
        application.mount(
            "/ui", StaticFiles(directory=str(frontend), html=True), name="ui"
        )

    @application.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse("/ui/kb.html" if (frontend / "kb.html").exists() else "/docs")

    return application


app = create_app()
