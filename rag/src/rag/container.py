"""Composition root.

One place that knows how to wire embeddings + vector store + LLM into the
services. The API, the CLI and the evaluator all build their dependencies
through here so behaviour stays identical across entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rag.config import settings
from rag.embeddings.base import EmbeddingService
from rag.embeddings.factory import build_embedding_service
from rag.llm.base import LLMService
from rag.llm.providers import build_judge_llm_service, build_llm_service
from rag.logging_utils import get_logger
from rag.services.generator import GenerationService
from rag.services.rag_service import RagService
from rag.services.retriever import RetrievalService
from rag.vectorstore.base import VectorStore

logger = get_logger(__name__)

#: Offline snapshot location for the in-memory store. Lives OUTSIDE rag/data
#: so that no vector database is ever written into the dataset directory.
OFFLINE_STORE_ENV = "RAG_OFFLINE_STORE"


@dataclass
class Container:
    embeddings: EmbeddingService
    store: VectorStore
    retriever: RetrievalService
    tenant: Optional[object] = None

    _llm: Optional[LLMService] = None
    _judge_llm: Optional[LLMService] = None

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = build_llm_service()
        return self._llm

    @property
    def judge_llm(self) -> LLMService:
        """The evaluator model - separate from the generator by design."""
        if self._judge_llm is None:
            self._judge_llm = build_judge_llm_service()
        return self._judge_llm

    def generator(self) -> GenerationService:
        return GenerationService(self.llm)

    def rag(self) -> RagService:
        return RagService(self.retriever, self.generator())

    def agentic(self, max_attempts: Optional[int] = None):
        """The LangGraph retrieve/evaluate/rewrite/generate workflow."""
        from rag.graph.workflow import AgenticRagWorkflow
        from rag.services.evaluator import LLMContextEvaluator, LLMQueryRewriter

        judge = self.judge_llm
        return AgenticRagWorkflow(
            retriever=self.retriever,
            generator=self.generator(),
            evaluator=LLMContextEvaluator(judge),
            rewriter=LLMQueryRewriter(judge),
            max_attempts=max_attempts,
        )

    def answerer(self, agentic: Optional[bool] = None, max_attempts: Optional[int] = None):
        """Whichever orchestrator is configured - both expose ``query()``."""
        use_graph = settings.AGENTIC_ENABLED if agentic is None else agentic
        if use_graph:
            workflow = self.agentic(max_attempts)

            class _GraphAdapter:
                """Gives the graph the same surface as :class:`RagService`."""

                def __init__(self, wf):
                    self._workflow = wf
                    self.retriever = wf.retriever
                    self.generator = wf.generator
                    self.evaluator = wf.evaluator
                    self.rewriter = wf.rewriter

                def query(self, question, top_k=None, filters=None):
                    return self._workflow.run(question, top_k=top_k, filters=filters)

            return _GraphAdapter(workflow)
        return self.rag()


    def kb(self, chunking=None):
        """Knowledge-base service for this container's tenant."""
        from rag.services.kb_service import KnowledgeBaseService
        from rag.services.tenancy import require_tenant

        return KnowledgeBaseService(
            require_tenant(self.tenant), self.store, self.embeddings, chunking
        )


def build_tenant_container(
    company_id: int,
    offline: bool = False,
    offline_path: Optional[Path] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> "Container":
    """Container scoped to one company's own collection."""
    from rag.services.tenancy import Tenant

    tenant = Tenant(company_id=company_id)
    embeddings = build_embedding_service(
        provider=embedding_provider, model=embedding_model
    )
    store = build_vector_store(
        embeddings, offline=offline, offline_path=offline_path,
        collection=tenant.collection_name,
    )
    container = Container(embeddings=embeddings, store=store,
                          retriever=None, tenant=tenant)  # type: ignore[arg-type]
    translator = None
    if settings.QUERY_TRANSLATION:
        from rag.services.query_translator import QueryTranslator

        translator = QueryTranslator(container.llm)
    container.retriever = RetrievalService(
        store, embeddings, tenant=tenant, translator=translator
    )
    return container


def build_vector_store(
    embeddings: EmbeddingService,
    offline: bool = False,
    offline_path: Optional[Path] = None,
    collection: Optional[str] = None,
) -> VectorStore:
    """Chroma Cloud by default; the in-memory store only when asked explicitly."""
    if offline:
        from rag.vectorstore.memory_store import InMemoryVectorStore

        name = collection or settings.CHROMA_COLLECTION_NAME
        if offline_path:
            logger.warning(
                "Using OFFLINE in-memory vector store at %s - not Chroma Cloud.",
                offline_path,
            )
            return InMemoryVectorStore.load(
                Path(offline_path), collection_name=name, embedding_model=embeddings.model
            )
        logger.warning("Using OFFLINE in-memory vector store - not Chroma Cloud.")
        return InMemoryVectorStore(collection_name=name, embedding_model=embeddings.model)

    from rag.vectorstore.chroma_cloud import ChromaCloudStore

    return ChromaCloudStore(collection_name=collection, embedding_model=embeddings.model)


def build_container(
    offline: bool = False,
    offline_path: Optional[Path] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    collection: Optional[str] = None,
) -> Container:
    embeddings = build_embedding_service(
        provider=embedding_provider, model=embedding_model
    )
    store = build_vector_store(
        embeddings, offline=offline, offline_path=offline_path, collection=collection
    )
    return Container(
        embeddings=embeddings,
        store=store,
        retriever=RetrievalService(store, embeddings),
    )


@lru_cache
def get_container() -> Container:
    """Cached container for the FastAPI app."""
    return build_container()
