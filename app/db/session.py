"""Database engine and session factory.

Supports two backends from one configuration:

* **PostgreSQL / Supabase** - set ``DATABASE_URL``. This is the path that works
  from macOS with no ODBC driver and no Windows integrated authentication.
* **SQL Server** - set ``DB_SERVER`` / ``DB_NAME`` / ``DB_DRIVER``. Uses
  username/password when ``DB_USER`` is provided, otherwise falls back to
  ``Trusted_Connection`` so existing Windows setups keep working unchanged.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from app.core.config import settings


def _build_url() -> URL | str:
    if settings.DATABASE_URL:
        url = settings.DATABASE_URL
        # SQLAlchemy needs an explicit driver; Supabase hands out postgres:// URLs.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    if not settings.DB_SERVER:
        raise RuntimeError(
            "No database configured. Set DATABASE_URL (PostgreSQL/Supabase) or "
            "DB_SERVER/DB_NAME/DB_DRIVER (SQL Server) in .env"
        )

    import pyodbc  # imported lazily so PostgreSQL users need no ODBC driver

    pyodbc.pooling = False
    # Integrated auth cannot work off Windows, so credentials are used when given.
    auth = (
        f"UID={settings.DB_USER};PWD={settings.DB_PASSWORD};"
        if settings.DB_USER
        else "Trusted_Connection=yes;"
    )
    odbc = (
        f"DRIVER={{{settings.DB_DRIVER}}};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"{auth}"
        "TrustServerCertificate=yes;"
    )
    return URL.create("mssql+pyodbc", query={"odbc_connect": odbc})


engine = create_engine(_build_url(), pool_pre_ping=True, echo=False)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)
