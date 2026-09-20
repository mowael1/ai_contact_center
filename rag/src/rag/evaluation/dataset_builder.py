"""Derives a curated evaluation set from real dataset content.

Ground truth is taken from the corpus itself: an FAQ accordion whose heading
is a genuine question becomes a question whose relevant source URL and section
path are known exactly. Nothing is invented - ``reference_answer`` is only ever
set to text that actually appears in the document.

Hand-written questions (paraphrases, mixed-language, out-of-scope) are layered
on top in :data:`CURATED_SEEDS`; each of those still points at a real URL, and
questions whose URL is absent from the corpus are dropped rather than kept with
a fabricated target.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from rag.evaluation.dataset import EvalDataset, EvalQuestion
from rag.htmlx.section_extractor import SectionExtractor
from rag.logging_utils import get_logger
from rag.services.ingest_service import iter_documents
from rag.text_utils import detect_language, short_hash

logger = get_logger(__name__)

# A heading is question-shaped when it asks something in Arabic or English.
_QUESTION_MARK = re.compile(r"[?؟]\s*$")
_EN_QUESTION_START = re.compile(
    r"^(how|what|can|do|does|is|are|where|when|why|who|which|will)\b", re.I
)
_AR_QUESTION_START = re.compile(r"^(ازاي|إزاي|ايه|إيه|هل|امتى|إمتى|فين|ليه|مين|كام|ما هي|ما هو)")


def looks_like_question(text: str) -> bool:
    text = (text or "").strip()
    if not (8 <= len(text) <= 160):
        return False
    return bool(
        _QUESTION_MARK.search(text)
        or _EN_QUESTION_START.match(text)
        or _AR_QUESTION_START.match(text)
    )


#: Hand-written questions covering categories the FAQ mining does not reach.
#: ``url`` must exist in the corpus or the entry is dropped.
CURATED_SEEDS: list[dict] = [
    {
        "question": "ازاي أجدد باقة الفليكس بتاعتي؟",
        "url": "https://web.vodafone.com.eg/ar/vodafone-flex",
        "category": "arabic_faq",
    },
    {
        "question": "How do I renew my Flex bundle?",
        "url": "https://web.vodafone.com.eg/en/vodafone-flex",
        "category": "english_faq",
    },
    {
        "question": "عايز أعرف إيه هي مزايا باقة RED الجديدة؟",
        "url": "https://web.vodafone.com.eg/ar/vodafone-red1",
        "category": "arabic_factual",
    },
    {
        "question": "What are the benefits of the new RED plans?",
        "url": "https://web.vodafone.com.eg/en/vodafone-red1",
        "category": "english_factual",
    },
    {
        "question": "ايه هو Vodafone Cash وازاي أستخدمه؟",
        "url": "https://web.vodafone.com.eg/ar/home",
        "category": "mixed_language",
    },
    {
        "question": "What is Vodafone DSL and what speeds are available?",
        "url": "https://web.vodafone.com.eg/en/home",
        "category": "english_factual",
    },
    {
        "question": "ازاي أشترك في كارت الفكة؟",
        "url": "https://web.vodafone.com.eg/ar/kart-el-korout",
        "category": "arabic_factual",
    },
    {
        "question": "Tell me about Fakka cards and how to recharge them",
        "url": "https://web.vodafone.com.eg/en/kart-el-korout",
        "category": "english_factual",
    },
]

#: Questions the knowledge base genuinely cannot answer. Used to measure
#: refusal / insufficient-context handling. They deliberately have no
#: retrieval ground truth.
OUT_OF_SCOPE: list[dict] = [
    {
        "question": "ما هو سعر سهم فودافون مصر في البورصة النهاردة؟",
        "category": "out_of_scope",
    },
    {
        "question": "What is the home address of the Vodafone Egypt CEO?",
        "category": "out_of_scope",
    },
    {
        "question": "كام موظف شغال في فروع فودافون في أسوان دلوقتي؟",
        "category": "out_of_scope",
    },
]


def _balance_by_language(
    candidates: list[EvalQuestion], max_total: int
) -> list[EvalQuestion]:
    """Round-robin across language buckets so no single language dominates."""
    buckets: dict[str, list[EvalQuestion]] = {}
    for question in candidates:
        buckets.setdefault(question.language, []).append(question)
    selected: list[EvalQuestion] = []
    while len(selected) < max_total and any(buckets.values()):
        for language in sorted(buckets):
            if len(selected) >= max_total:
                break
            if buckets[language]:
                selected.append(buckets[language].pop(0))
    return selected


def build_evaluation_dataset(
    file: Optional[Path] = None,
    directory: Optional[Path] = None,
    limit: int = 600,
    max_mined: int = 40,
) -> EvalDataset:
    extractor = SectionExtractor()
    questions: list[EvalQuestion] = []
    candidates: list[EvalQuestion] = []
    seen_questions: set[str] = set()

    # url -> (document_id, {section_path_lower: section})
    index: dict[str, tuple[str, dict[str, object]]] = {}

    for document in iter_documents(file=file, directory=directory, limit=limit):
        sections = extractor.extract(document.raw_html, document.retrieval_text)
        if not sections:
            continue
        index[document.source_url] = (
            document.document_id,
            {s.path_string: s for s in sections},
        )

        # Mine FAQ accordions: the heading is a real question, the panel is a
        # real answer, so both sides of the ground truth come from the corpus.
        for section in sections:
            if section.extraction_method != "html_component":
                continue
            title = (section.section_title or "").strip()
            if not looks_like_question(title) or title.lower() in seen_questions:
                continue
            if len(section.text) < 80:
                continue
            seen_questions.add(title.lower())
            candidates.append(
                EvalQuestion(
                    id=f"faq-{short_hash(document.source_url, title, length=10)}",
                    question=title,
                    category="mined_faq",
                    language=detect_language(title),
                    # Verbatim page text - not a generated answer.
                    reference_answer=section.text[:900],
                    relevant_source_urls=[document.source_url],
                    relevant_document_ids=[document.document_id],
                    relevant_sections=[section.path_string],
                    expected_citations=[document.source_url],
                    graded_relevance={document.source_url: 3.0},
                    expect_insufficient=False,
                    notes="Mined from an FAQ accordion; reference answer is verbatim page text.",
                )
            )

    # Balance the mined pool across languages: the corpus has far more English
    # FAQ accordions than Arabic ones, and taking the first N would make the
    # evaluation set almost monolingual.
    questions.extend(_balance_by_language(candidates, max_mined))

    # Curated questions, kept only when their target URL exists in the corpus.
    dropped: list[str] = []
    for seed in CURATED_SEEDS:
        entry = index.get(seed["url"])
        if entry is None:
            dropped.append(seed["url"])
            continue
        document_id, sections = entry
        questions.append(
            EvalQuestion(
                id=f"curated-{short_hash(seed['question'], length=10)}",
                question=seed["question"],
                category=seed["category"],
                language=detect_language(seed["question"]),
                relevant_source_urls=[seed["url"]],
                relevant_document_ids=[document_id],
                relevant_sections=[],
                expected_citations=[seed["url"]],
                graded_relevance={seed["url"]: 3.0},
                expect_insufficient=False,
                notes="Hand-written question against a verified corpus URL.",
            )
        )
    if dropped:
        logger.warning("Dropped %d curated questions with URLs absent from corpus", len(dropped))

    for seed in OUT_OF_SCOPE:
        questions.append(
            EvalQuestion(
                id=f"oos-{short_hash(seed['question'], length=10)}",
                question=seed["question"],
                category=seed["category"],
                language=detect_language(seed["question"]),
                expect_insufficient=True,
                notes="Answer is not in the knowledge base; measures refusal behaviour.",
            )
        )

    logger.info(
        "Built evaluation dataset: %d questions (%d mined, %d curated, %d out-of-scope)",
        len(questions), sum(1 for q in questions if q.category == "mined_faq"),
        len(CURATED_SEEDS) - len(dropped), len(OUT_OF_SCOPE),
    )
    return EvalDataset(
        name="vf_egypt_eval",
        description=(
            "Evaluation set for the Vodafone Egypt RAG knowledge base. FAQ "
            "questions and reference answers are mined verbatim from the "
            "scraped pages; curated questions target verified corpus URLs; "
            "out-of-scope questions carry no retrieval ground truth and exist "
            "to measure insufficient-context handling."
        ),
        questions=questions,
    )
