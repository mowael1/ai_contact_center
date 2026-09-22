from fastapi import APIRouter

from app.api.routes import (
    auth,
    calls,
    companies,
    customers,
    tickets,
    users,
    vonage,
)


api_router = APIRouter()


api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)


api_router.include_router(
    companies.router,
    prefix="/companies",
    tags=["Companies"]
)


api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

api_router.include_router(
    customers.router,
    prefix="/customers",
    tags=["Customers"]
)

api_router.include_router(
    tickets.router,
    prefix="/tickets",
    tags=["Tickets"]
)

api_router.include_router(
    calls.router,
    prefix="/calls",
    tags=["Calls"]
)

api_router.include_router(
    vonage.router,
    prefix="/vonage",
    tags=["Vonage"]
)

# Per-company knowledge base and agent assistant. Imported lazily so the API
# still starts if the RAG extras are not installed.
try:
    from rag.api.kb_routes import router as kb_router

    api_router.include_router(kb_router, prefix="/kb", tags=["Knowledge Base"])
except ImportError:  # pragma: no cover - optional component
    pass
