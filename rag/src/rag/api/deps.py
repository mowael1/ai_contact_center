"""Request dependencies - above all, resolving the tenant.

``company_id`` is never read from a request body. In ``jwt`` mode it comes from
the authenticated user's database row; in ``dev`` mode from the
``X-Company-Id`` header, so the knowledge base can be exercised before the
database is reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from rag.config import settings
from rag.logging_utils import get_logger
from rag.services.tenancy import Tenant

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """Everything this module needs about the caller, read eagerly.

    A detached ORM instance cannot lazy-load ``user.role`` once its session has
    closed, so the role name is resolved *inside* the session and carried out
    as plain values. Returning the ORM object instead raised
    ``DetachedInstanceError`` on the first role check.
    """

    user_id: Optional[int]
    company_id: Optional[int]
    role: str


def _current_user(request: Request) -> Optional[CallerIdentity]:
    """The authenticated caller in jwt mode, otherwise None.

    Imported lazily and resolved per request so the RAG service still runs
    standalone, without the main application or its database being importable.
    """
    if settings.RAG_AUTH_MODE != "jwt":
        return None
    try:
        from app.api.dependencies import get_current_user, get_db
        from app.db.session import SessionLocal
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG_AUTH_MODE=jwt requires the main application package",
        ) from exc

    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    with SessionLocal() as db:
        user = get_current_user(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=credential
            ),
            db=db,
        )
        # Read every attribute while the session is still open.
        role = getattr(user, "role", None)
        return CallerIdentity(
            user_id=getattr(user, "id", None),
            company_id=getattr(user, "company_id", None),
            role=(getattr(role, "name", "") or "").strip().lower(),
        )


#: Roles permitted to read a company they do not belong to. Delegation is
#: granted by role, never merely by having no company of one's own.
DELEGATING_ROLES = {"super_admin"}


def _role_name(user) -> str:
    """Role name from either a CallerIdentity or a raw ORM user (tests)."""
    role = getattr(user, "role", "")
    if isinstance(role, str):
        return role.strip().lower()
    return (getattr(role, "name", "") or "").strip().lower()


def get_tenant(
    x_company_id: Optional[int] = Header(
        None, alias="X-Company-Id",
        description=(
            "The company to act on. In jwt mode this is honoured only for a "
            "super admin choosing a company; for everyone else it is ignored "
            "and their own company is used."
        ),
    ),
    user=Depends(_current_user),
) -> Tenant:
    """Resolve the caller's tenant.

    A normal user always gets their own company; a header they send is ignored.
    A super admin belongs to no company, so they must name one explicitly -
    there is deliberately no "see everything" scope.
    """
    if settings.RAG_AUTH_MODE == "jwt":
        own_company = getattr(user, "company_id", None)
        if own_company:
            # A company's own member. Never overridable from the request.
            return Tenant(company_id=int(own_company))

        if _role_name(user) in DELEGATING_ROLES:
            if x_company_id is None:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Choose a company to view. A super admin has no "
                        "knowledge base of their own, and there is no "
                        "cross-company view."
                    ),
                )
            if x_company_id <= 0:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    detail="X-Company-Id must be a positive integer",
                )
            # Audited: reading another company's documents is a privileged act.
            logger.info(
                "Delegated access: user id=%s role=%s acting on company %s",
                getattr(user, "id", "?"), _role_name(user), x_company_id,
            )
            return Tenant(company_id=int(x_company_id), is_delegated=True)

        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=(
                "This account is not attached to a company, so it has no "
                "knowledge base to read."
            ),
        )

    # No silent fallback. Defaulting a missing header to some company id made
    # every caller resolve to the same tenant, which looks exactly like a
    # working system while showing one company's documents to everyone.
    if x_company_id is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                "X-Company-Id header is required while RAG_AUTH_MODE=dev. "
                "Set RAG_AUTH_MODE=jwt so the company is taken from the "
                "signed-in user instead."
            ),
        )
    if x_company_id <= 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="X-Company-Id must be positive"
        )
    return Tenant(company_id=int(x_company_id))


@lru_cache(maxsize=64)
def _cached_container(company_id: int):
    """One container per company, reused across requests.

    Building it opens a Chroma Cloud client and resolves the collection, which
    is a network round trip. Doing that per request added roughly a second to
    every answer.
    """
    from rag.container import build_tenant_container

    return build_tenant_container(company_id=company_id)


def get_kb_container(tenant: Tenant = Depends(get_tenant)):
    """A container scoped to the caller's own collection."""
    return _cached_container(tenant.company_id)
