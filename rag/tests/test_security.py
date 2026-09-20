"""Credentials must never reach logs, diagnostics or committed files."""

import logging

import pytest

from rag.config import Settings
from rag.logging_utils import RedactSecretsFilter, redact


@pytest.mark.parametrize(
    "text",
    [
        "key is sk-abcdef1234567890",
        "api_key: ck-secret9999999",
        "Authorization: Bearer tok_abcdef123456",
        "x-chroma-token=abcd1234efgh",
        "CHROMA_API_KEY=mysecretvalue123",
        "anthropic_api_key: sk-ant-abc123456789",
    ],
)
def test_secret_shaped_strings_are_redacted(text):
    assert "REDACTED" in redact(text)


@pytest.mark.parametrize(
    "secret", ["sk-abcdef1234567890", "ck-secret9999999", "mysecretvalue123", "abcd1234efgh"]
)
@pytest.mark.parametrize(
    "template", ["CHROMA_API_KEY={}", "x-chroma-token={}", "Bearer {}", "api_key: {}"]
)
def test_secret_value_never_survives_a_labelled_context(template, secret):
    """In any key-labelled context the value itself must be gone.

    Redaction is pattern-based defence in depth: a bare opaque string with no
    surrounding label (``"key is abcd1234efgh"``) is indistinguishable from
    ordinary prose and is deliberately not redacted. The primary guarantee is
    that the code never formats a credential into a log message at all.
    """
    assert secret not in redact(template.format(secret))


@pytest.mark.parametrize("secret", ["sk-abcdef1234567890", "ck-secret9999999"])
def test_prefixed_keys_are_caught_even_without_a_label(secret):
    assert secret not in redact(f"connecting using {secret} now")


def test_ordinary_log_lines_are_untouched():
    line = "Retrieved 5 chunks for query (24 chars) in 12.4 ms"
    assert redact(line) == line


def test_logging_filter_rewrites_the_record():
    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1, "connecting with sk-abcdef1234567890", (), None
    )
    RedactSecretsFilter().filter(record)
    assert "sk-abcdef1234567890" not in record.getMessage()


def _isolated(**kw) -> Settings:
    """Settings that ignore any real rag/.env, so tests stay hermetic."""
    return Settings(_env_file=None, **kw)


def test_safe_dump_masks_every_secret_field():
    settings = _isolated(
        CHROMA_API_KEY="sk-chromasecret123",
        OPENAI_API_KEY="sk-openaisecret123",
        ANTHROPIC_API_KEY="sk-anthropicsecret1",
        COHERE_API_KEY="cohere-secret-12345",
    )
    dumped = str(settings.safe_dump())
    for secret in ("chromasecret", "openaisecret", "anthropicsecret", "cohere-secret"):
        assert secret not in dumped
    assert settings.safe_dump()["CHROMA_API_KEY"].startswith("<set:")


def test_safe_dump_marks_unset_keys():
    assert _isolated(CHROMA_API_KEY="").safe_dump()["CHROMA_API_KEY"] == "<unset>"


def test_chroma_is_configured_requires_all_three():
    assert not _isolated(CHROMA_API_KEY="k").chroma_is_configured()
    assert not _isolated(CHROMA_API_KEY="k", CHROMA_TENANT="t").chroma_is_configured()
    assert _isolated(
        CHROMA_API_KEY="k", CHROMA_TENANT="t", CHROMA_DATABASE="d"
    ).chroma_is_configured()


def test_gemini_api_key_alias_is_accepted():
    """Google's docs use both GOOGLE_API_KEY and GEMINI_API_KEY."""
    from_gemini = Settings(_env_file=None, GEMINI_API_KEY="abc123")
    assert from_gemini.GOOGLE_API_KEY == "abc123"

    from_google = Settings(_env_file=None, GOOGLE_API_KEY="xyz789")
    assert from_google.GEMINI_API_KEY == "xyz789"


def test_gemini_key_is_redacted_in_safe_dump():
    dumped = str(Settings(_env_file=None, GEMINI_API_KEY="supersecretgeminikey").safe_dump())
    assert "supersecretgeminikey" not in dumped
