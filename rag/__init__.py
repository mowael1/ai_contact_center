"""Development-time bridge for the RAG package's ``src`` layout.

The contact-center application is commonly run directly from the repository
root (``uvicorn app.main:app``). In that mode Python sees this outer ``rag``
directory, while the importable package itself lives in ``rag/src/rag``.
Extending the package path keeps that normal command working without requiring
callers to remember a custom ``PYTHONPATH``.

Installed builds still use the regular package from ``rag/src``.
"""

from pathlib import Path


_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src" / "rag"

if _SOURCE_PACKAGE.is_dir():
    __path__.append(str(_SOURCE_PACKAGE))
