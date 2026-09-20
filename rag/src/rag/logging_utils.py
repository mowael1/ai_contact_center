"""Logging helpers.

A single ``configure_logging`` entry point plus a redaction filter that keeps
API keys and tokens out of the log stream even if a caller accidentally
formats one into a message.
"""

from __future__ import annotations

import logging
import re
import sys

# Order matters: "Bearer <token>" and named-header forms are handled before the
# generic Authorization rule, so the token itself is scrubbed rather than just
# the scheme word.
_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(ck-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(?i)(bearer\s+)(\S{4,})"),
    re.compile(
        r"(?i)((?:x-)?(?:chroma|anthropic|openai|cohere)?[_-]?"
        r"(?:api[_-]?key|token|secret)\"?\s*[:=]\s*\"?)([A-Za-z0-9_\-.]{4,})"
    ),
    re.compile(r"(?i)(authorization\"?\s*[:=]\s*\"?)(\S{4,})"),
]


class RedactSecretsFilter(logging.Filter):
    """Scrubs anything that looks like a credential from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def redact(text: str) -> str:
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 1:
            out = pattern.sub("***REDACTED***", out)
        else:
            out = pattern.sub(lambda m: m.group(1) + "***REDACTED***", out)
    return out


def configure_logging(level: str | None = None) -> None:
    from rag.config import settings

    resolved = (level or settings.LOG_LEVEL).upper()
    root = logging.getLogger()
    root.setLevel(resolved)
    if any(getattr(h, "_rag_handler", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    )
    handler.addFilter(RedactSecretsFilter())
    handler._rag_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
