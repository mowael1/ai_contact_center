"""Centralised configuration for the RAG backend.

Every tunable lives here and is sourced from environment variables so that no
credential is ever hard-coded. Follows the ``pydantic-settings`` convention
already used by ``app/core/config.py`` in the parent project.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RAG_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = RAG_ROOT / "data"
DEFAULT_EVAL_DIR = DEFAULT_DATA_DIR / "evaluation"

SECRET_FIELDS = {
    "CHROMA_API_KEY",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "HF_API_TOKEN",
}


class Settings(BaseSettings):
    # ---- Data -------------------------------------------------------------
    DATA_DIR: Path = DEFAULT_DATA_DIR
    EVALUATION_DIR: Path = DEFAULT_EVAL_DIR

    # ---- Chroma Cloud -----------------------------------------------------
    CHROMA_API_KEY: str = ""
    CHROMA_TENANT: str = ""
    CHROMA_DATABASE: str = ""
    CHROMA_COLLECTION_NAME: str = "vf_egypt_kb"

    # ---- Embeddings -------------------------------------------------------
    # provider: openai | cohere | sentence_transformers | hash
    EMBEDDING_PROVIDER: Literal[
        "embeddinggemma_hf", "embeddinggemma_local", "openai", "cohere",
        "sentence_transformers", "local_lexical", "hash",
    ] = "embeddinggemma_hf"
    EMBEDDING_MODEL: str = "google/embeddinggemma-300m"
    EMBEDDING_BATCH_SIZE: int = 96
    EMBEDDING_CACHE_ENABLED: bool = True
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    COHERE_API_KEY: str = ""
    HF_API_TOKEN: str = ""
    HF_INFERENCE_ENDPOINT: str = "https://router.huggingface.co/hf-inference"

    # ---- LLM --------------------------------------------------------------
    # provider: gemini | anthropic | openai | echo (offline stub, never fabricates)
    LLM_PROVIDER: Literal["gemini", "anthropic", "openai", "echo"] = "gemini"
    LLM_MODEL: str = "gemini-2.5-pro"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0
    LLM_STREAMING: bool = True
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # ---- Judge / evaluator LLM -------------------------------------------
    # A separate, cheaper model grades retrieved context inside the agentic
    # graph, so the generator never decides on its own whether it had enough
    # to work with.
    JUDGE_LLM_PROVIDER: Literal["gemini", "anthropic", "openai", "echo"] = "gemini"
    JUDGE_LLM_MODEL: str = "gemini-2.5-flash"

    # ---- Agentic retrieval graph -----------------------------------------
    # Linear pipeline when false; LangGraph retrieve/evaluate/rewrite loop when true.
    AGENTIC_ENABLED: bool = True
    #: Total retrieval attempts, including the first. 2 means one rewrite.
    MAX_RETRIEVAL_ATTEMPTS: int = 3
    #: Minimum judge confidence for context to count as relevant.
    CONTEXT_RELEVANCE_THRESHOLD: float = 0.5

    # ---- Retrieval-text policy -------------------------------------------
    # enhanced_first  -> always prefer llm_enhanced_text (spec default)
    # language_aware  -> prefer llm_enhanced_text only when its script matches
    #                    raw_text (keeps Arabic pages Arabic)
    # raw_only        -> always use raw_text
    RETRIEVAL_TEXT_POLICY: Literal[
        "enhanced_first", "language_aware", "raw_only"
    ] = "enhanced_first"

    # ---- Chunking ---------------------------------------------------------
    CHUNK_TARGET_SIZE: int = 700
    CHUNK_MIN_SIZE: int = 200
    CHUNK_MAX_SIZE: int = 1400
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.55
    CHUNK_OVERLAP: int = 1  # measured in sentences
    MIN_SECTION_CHARS: int = 40

    # ---- Retrieval --------------------------------------------------------
    TOP_K: int = 5
    #: Collapse retrieved passages with identical text (same content published
    #: under several URLs). Keeps the best-scoring copy.
    DEDUPLICATE_RESULTS: bool = True
    MAX_CONTEXT_CHARS: int = 8000

    # ---- Ingestion --------------------------------------------------------
    INGEST_BATCH_SIZE: int = 128
    SUCCESS_STATUS: str = "success"

    # ---- Logging ----------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=(RAG_ROOT / ".env", ".env"),
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("CHUNK_MAX_SIZE")
    @classmethod
    def _max_gt_target(cls, v: int, info) -> int:
        target = info.data.get("CHUNK_TARGET_SIZE")
        if target and v < target:
            raise ValueError("CHUNK_MAX_SIZE must be >= CHUNK_TARGET_SIZE")
        return v

    # -- helpers ------------------------------------------------------------
    def chroma_is_configured(self) -> bool:
        return bool(self.CHROMA_API_KEY and self.CHROMA_TENANT and self.CHROMA_DATABASE)

    def safe_dump(self) -> dict:
        """Settings with every secret redacted - safe to log or print."""
        out = {}
        for key, value in self.model_dump().items():
            if key in SECRET_FIELDS:
                out[key] = f"<set:{len(str(value))} chars>" if value else "<unset>"
            else:
                out[key] = str(value) if isinstance(value, Path) else value
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
