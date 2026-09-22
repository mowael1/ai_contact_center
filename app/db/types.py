"""Portable column types.

The models were written against SQL Server (``DATETIME2``, ``SYSUTCDATETIME()``,
``NVARCHAR`` via ``Unicode``). Supabase is PostgreSQL, so these definitions map
each construct to whatever the configured dialect supports, and the models
import from here instead of from ``sqlalchemy.dialects.mssql``.

Nothing here is dialect-specific at import time: the choice happens when
SQLAlchemy compiles the statement for the connected database, so the same
models run on either backend.
"""

from __future__ import annotations

from sqlalchemy import DateTime, TypeDecorator
from sqlalchemy.dialects.mssql import DATETIME2
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import expression


class UtcDateTime(TypeDecorator):
    """``DATETIME2`` on SQL Server, ``TIMESTAMP`` on PostgreSQL.

    Replaces the direct ``mssql.DATETIME2`` import so the models are no longer
    tied to one database.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "mssql":
            return dialect.type_descriptor(DATETIME2())
        if dialect.name == "postgresql":
            return dialect.type_descriptor(TIMESTAMP(timezone=False))
        return dialect.type_descriptor(DateTime())


class utcnow(expression.FunctionElement):
    """Server-side "now in UTC", compiled per dialect.

    Used as ``server_default=utcnow()`` in place of the raw
    ``text("SYSUTCDATETIME()")``, which only PostgreSQL would reject.
    """

    type = DateTime()
    inherit_cache = True


from sqlalchemy.ext.compiler import compiles  # noqa: E402


@compiles(utcnow, "mssql")
def _utcnow_mssql(element, compiler, **kw):
    return "SYSUTCDATETIME()"


@compiles(utcnow, "postgresql")
def _utcnow_postgresql(element, compiler, **kw):
    return "(NOW() AT TIME ZONE 'utc')"


@compiles(utcnow)
def _utcnow_default(element, compiler, **kw):
    return "CURRENT_TIMESTAMP"
