"""Evaluation dataset schema and loading.

Ground truth is optional per field. Anything absent makes the dependent metric
"unavailable" rather than zero - see :mod:`rag.evaluation.runner`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from rag.config import settings


class EvalQuestion(BaseModel):
    id: str
    question: str
    category: str = "general"
    language: str = "unknown"
    reference_answer: Optional[str] = None
    relevant_source_urls: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_sections: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    #: url -> graded relevance (0..3). Enables graded nDCG when present.
    graded_relevance: dict[str, float] = Field(default_factory=dict)
    #: True when the KB is expected NOT to contain the answer.
    expect_insufficient: Optional[bool] = None
    notes: str = ""

    def has_retrieval_ground_truth(self) -> bool:
        return bool(
            self.relevant_source_urls or self.relevant_document_ids or self.relevant_sections
        )


class EvalDataset(BaseModel):
    name: str = "vf_egypt_eval"
    description: str = ""
    questions: list[EvalQuestion] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EvalDataset":
        target = Path(path) if path else settings.EVALUATION_DIR / "retrieval_eval.json"
        if not target.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {target}. "
                "Generate one with: rag-cli eval build-dataset"
            )
        data: Any = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, list):
            data = {"questions": data}
        return cls.model_validate(data)

    def sample(self, limit: int) -> "EvalDataset":
        """A balanced subset, round-robin across categories.

        Useful when an LLM provider's daily quota is smaller than the full set -
        a naive head(n) would evaluate only one category.
        """
        if limit >= len(self.questions):
            return self
        buckets: dict[str, list[EvalQuestion]] = {}
        for question in self.questions:
            buckets.setdefault(question.category, []).append(question)
        picked: list[EvalQuestion] = []
        while len(picked) < limit and any(buckets.values()):
            for key in sorted(buckets):
                if len(picked) >= limit:
                    break
                if buckets[key]:
                    picked.append(buckets[key].pop(0))
        return EvalDataset(name=self.name, description=self.description, questions=picked)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
