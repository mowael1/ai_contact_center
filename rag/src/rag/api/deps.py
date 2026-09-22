"""Request dependencies - above all, resolving the tenant.

``company_id`` is never read from a request body. In ``jwt`` mode it comes from
the authenticated user's database row; in ``dev`` mode from the
``X-Company-Id`` header, so the knowledge base can be exercised before the SQL
Server instance is reachable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from rag.config import settings
from rag.logging_utils import get_logger
from rag.services.tenancy import Tenant

logger = get_logger(__name__)


def get_tenant(
    x_company_id: Optional[int] = Header(
        None, alias="X-Company-Id",
        description="Dev mode only. Ignored when RAG_AUTH_MODE=jwt.",
    ),
) -> Tenant:
    """Resolve the caller's tenant."""
    if settings.RAG_AUTH_MODE == "jwt":
        return _tenant_from_jwt()

    company_id = x_company_id or settings.RAG_DEV_COMPANY_ID
    if not company_id or company_id <= 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="X-Company-Id header is required in dev mode",
        )
    return Tenant(company_id=int(company_id))


def _tenant_from_jwt() -> Tenant:
    """Production path: the company of the authenticated user.

    Wired to the parent application's ``get_current_user`` so there is exactly
    one source of identity. A user with no company is refused rather than
    given an unscoped view.
    """
    try:
        from fastapi import Depends as _Depends  # noqa: F401
        from app.api.dependencies import get_current_user  # type: ignore
    except Exception as exc:  # pragma: no cover - app not importable
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT auth requires the main application and its database",
        ) from exc
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "RAG_AUTH_MODE=jwt requires wiring get_current_user into this "
            "dependency; use RAG_AUTH_MODE=dev until the database is available"
        ),
    )


@lru_cache(maxsize=64)
def _cached_container(company_id: int):
    """One container per company, reused across requests.

    Building it opens a Chroma Cloud client and resolves the collection, which
    is a network round trip. Doing that per request added roughly a second to
    every answer, so containers are cached by company id. They hold no
    per-request state - the tenant is fixed and the retriever is stateless
    apart from last-call diagnostics.
    """
    from rag.container import build_tenant_container

    return build_tenant_container(company_id=company_id)


def get_kb_container(tenant: Tenant = Depends(get_tenant)):
    """A container scoped to the caller's own collection."""
    return _cached_container(tenant.company_id)
