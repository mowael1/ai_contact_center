"""Tenant scoping.

Isolation rule: a company may only ever read its own documents. This is
enforced two independent ways, so both must fail before a leak is possible:

1. **A separate Chroma collection per company** (``kb_company_{id}``). If a
   scope is ever lost, the query hits an empty/absent collection and returns
   nothing - the failure direction is "no results", never "everyone's results".
2. **A mandatory ``company_id`` metadata filter** inside that collection.

``company_id`` is resolved from the authenticated user server-side. A value in
a request body is never trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

COLLECTION_PREFIX = "kb_company_"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]$")


class TenantScopeError(PermissionError):
    """Raised when an operation is attempted without a company scope.

    Deliberately an error rather than a silent fallback to an unscoped search.
    """


@dataclass(frozen=True, slots=True)
class Tenant:
    company_id: int
    #: True for a super admin acting on a company they do not belong to.
    is_delegated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.company_id, int) or self.company_id <= 0:
            raise TenantScopeError(
                f"company_id must be a positive integer, got {self.company_id!r}"
            )

    @property
    def collection_name(self) -> str:
        return collection_for(self.company_id)


def collection_for(company_id: Optional[int]) -> str:
    """Chroma collection name for a company.

    Chroma requires 3-63 chars of [A-Za-z0-9_-] starting and ending
    alphanumeric, which ``kb_company_<int>`` always satisfies.
    """
    if company_id is None:
        raise TenantScopeError("cannot build a collection name without a company_id")
    if not isinstance(company_id, int) or company_id <= 0:
        raise TenantScopeError(f"invalid company_id: {company_id!r}")
    name = f"{COLLECTION_PREFIX}{company_id}"
    if not _SAFE.match(name):  # pragma: no cover - unreachable for valid ints
        raise TenantScopeError(f"unsafe collection name: {name!r}")
    return name


def require_tenant(tenant: Optional[Tenant]) -> Tenant:
    if tenant is None:
        raise TenantScopeError(
            "this operation requires a company scope; refusing an unscoped query"
        )
    return tenant
