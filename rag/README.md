# Vodafone Egypt RAG Backend

Arabic/English retrieval-augmented generation over a web-scraped Vodafone Egypt
knowledge base. Backend only — Python services, a REST API and a CLI. There is
no UI in this project by design.

---

## 1. Purpose

Give customer-service agents grounded, citable answers — in Egyptian Arabic,
English, or a mix — drawn strictly from the scraped Vodafone Egypt public web
pages. The system never answers from model memory: if the knowledge base does
not contain the information, it says so.

## 2. Architecture

```text
Parquet
  ↓
Document Ingestion
  ↓
raw_html
  ↓
HTML Section Extraction
  ↓
Sentence Segmentation
  ↓
Semantic Chunking
  ↓
Embeddings
  ↓
Chroma Cloud
  ↓
Retriever
  ↓
Context Builder
  ↓
LLM Generator
  ↓
Citation Builder
  ↓
FastAPI / CLI
```

Each arrow is a separate module with a single responsibility. There is no
god-function: every stage can be imported and run on its own.

## 3. Folder structure

```text
rag/
├── data/                        # the ONLY data directory
│   ├── batch_0000.parquet ...   # 29 files, directly under data/
│   └── evaluation/
│       ├── retrieval_eval.json          # curated evaluation dataset
│       ├── retrieval_results.json       # stage 2 report
│       ├── rag_results.json             # stage 3-6 report
│       ├── chunking_results.json        # stage 1 report
│       ├── dataset_report.json
│       ├── extraction_report.json
│       ├── evaluation_summary.json
│       └── evaluation_summary.md
├── src/rag/
│   ├── config.py                # all settings, env-sourced
│   ├── logging_utils.py         # logging + secret redaction
│   ├── models.py                # Document / Section / Chunk / RetrievedChunk
│   ├── text_utils.py            # language detection, normalisation, hashing
│   ├── container.py             # composition root
│   ├── cli.py
│   ├── ingestion/               # parquet_loader, document_builder
│   ├── htmlx/                   # cleaner, components, section_extractor
│   ├── chunking/                # sentence_segmenter, semantic_chunker
│   ├── embeddings/              # base, hosted_providers, local_providers, factory
│   ├── vectorstore/             # base, chroma_cloud, memory_store
│   ├── llm/                     # base, providers
│   ├── services/                # retriever, context_builder, generator,
│   │                            # citations, rag_service, ingest_service,
│   │                            # interfaces, prompts
│   ├── evaluation/              # metrics, dataset, dataset_builder,
│   │                            # diagnostics, stages, runner
│   └── api/                     # app, routes, schemas
├── tests/
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

No ChromaDB is stored on disk. There is no `data/raw/`, `data/processed/` or
`data/chroma/`.

## 4. Dataset structure

29 Parquet files, 2 804 rows, ~24 MB. One row per scraped page:

| field | notes |
|---|---|
| `source_url` | unique across the corpus (no duplicate URLs) |
| `scrape_status` | `success` (2 591), `failed:timeout` (126), `failed:nav:Error` (87) |
| `raw_html` | median 122 KB, max 2.5 MB — full SPA markup |
| `raw_text` | median 1 053 chars |
| `llm_enhanced_text` | missing for 580 rows |
| `llm_model` / `prompt_version` | `qwen2.5-coder:latest` / `v5.0` |
| `scraped_at`, `llm_processed_at`, `processing_time_sec` | scrape metadata |

### Dataset findings that shaped the implementation

These were measured, not assumed:

1. **`scrape_status` is `success`**, not `successful`.
2. **~15% of `success` rows are soft-404s** — the page fetched fine but renders
   the not-found template (`لا يوجد صفحة مسجلة بهذا العنوان`). They are filtered
   out at ingestion; otherwise one template becomes ~100 near-identical chunks.
3. **88% of Arabic pages have an English `llm_enhanced_text`.** The enhancement
   step translated them. This matters only for the text-fallback path, because
   chunk text normally comes from `raw_html` (original language). See
   `RETRIEVAL_TEXT_POLICY`.
4. **The HTML is a Liferay portal plus a bespoke component library.** Heading
   tags are used properly (h3 is the most common), and `.journal-content-article`
   is the reliable content wrapper.
5. **Accordions appear in two populations.** Navigation menus and editorial FAQs
   share the same `js-accordion-*` classes, so a naive "buttons are headings"
   rule would fill the index with menu labels. See §6.
6. **Owl Carousel clones every slide**, duplicating the same passage 3-5× per
   page. `.cloned` nodes are dropped.
7. **The same content is published under several URLs** (`/en/vodafone-cash` and
   `/en/Vodafone-cash`, `/ar/recharge-bill-payment` and `/recharge-bill-payment`).
   Handled by deduplicating at retrieval time rather than discarding URLs.

## 5. HTML-first section extraction

`raw_html` is the **primary** structural source. No LLM is involved.

1. **Clean** (`htmlx/cleaner.py`) — drop `script`/`style`/`svg`/`form`…, drop
   `nav`/`footer`/`header`/`aside`, drop anything whose class or id matches the
   observed chrome patterns (`navigation`, `footer-list`, `cookie`, `cloned`, …)
   or whose ARIA role is `navigation`/`banner`/`contentinfo`.
2. **Pick the content root** — the densest of `main`, `[role=main]`,
   `.journal-content-article`, `article`, `#content`, measured in text length so
   an empty `<main>` shell cannot beat the real article. Multiple article blocks
   resolve to their lowest common ancestor.
3. **Walk in DOM order**, opening a section at each `h1`–`h4` and attaching the
   paragraphs, lists and tables that follow. A heading stack yields the full
   hierarchy:

   ```json
   {
     "section_title": "Renewal",
     "section_path": ["Internet", "Internet Packages", "Renewal"],
     "heading_level": 3
   }
   ```

   Tables render as `cell | cell` rows and lists as `- item`, so structure
   survives into the chunk text. `h5`/`h6` are content by default and are only
   promoted to headings in documents that have no `h1`–`h4` at all.

### Extraction methods

Every section records how it was found:

| method | meaning |
|---|---|
| `html_structure` | from a real heading tag |
| `html_component` | from an accordion / tab / FAQ control |
| `text_fallback` | deterministic text heuristics (HTML gave nothing) |
| `no_section` | no reliable heading — `section_title` and `section_path` are `null` |

A null title is always preferred over an invented one.

## 6. Accordions, tabs and FAQs

A control becomes a section only if it clears every hard gate:

- it is not inside navigation chrome and has no nav-like class;
- its `aria-controls` / `data-target` / `href="#id"` resolves to a real element;
- that panel holds ≥ 60 characters of prose and does not merely echo the label.

Survivors are then scored (content-accordion classes, being an `h*` tag, having
`aria-controls`, matching `aria-labelledby`, panel length) and must reach 0.6.

This is what separates the two populations found in the corpus:

```html
<!-- Navigation: rejected -->
<a class="js-accordion-heading js-navigation-link" role="tab" aria-expanded="false">Shop</a>

<!-- Editorial FAQ: accepted as a section -->
<h3 class="accordion-title js-collapser" id="heading-0" aria-controls="collapse-0">
  <span class="accordion-title-heading">Renew Your Bundle</span>
</h3>
<div id="collapse-0" aria-labelledby="heading-0">
  <div class="card-body"><p>You can renew your bundle … dial *880#.</p></div>
</div>
```

Panels rendered outside the chosen content root (common for tab panels) are
added explicitly so their content is not lost.

## 7. Semantic chunking

No LLM. Sentence embeddings and cosine similarity only.

```text
Section → Sentences → Sentence embeddings → Neighbour similarity
        → Semantic boundaries → Size constraints → Chunks
```

**The section is a hard boundary.** Chunks are built strictly inside one
section; two adjacent sections never merge no matter how similar their
sentences are. A section that fits within `CHUNK_MAX_SIZE` stays whole.

Algorithm:

1. Split into sentences (§8).
2. Hard-wrap any single sentence longer than `CHUNK_MAX_SIZE` (the corpus has
   3 000-character run-on T&C blocks).
3. Embed all sentences of the document in one batch.
4. Cosine similarity between each adjacent pair.
5. Propose a boundary where similarity drops — below the configured
   `SEMANTIC_SIMILARITY_THRESHOLD` **or** below `mean − stdev` for that section,
   so uniformly-dense and uniformly-diverse sections are both handled.
6. Accumulate sentences; **commit** a proposed boundary only once the chunk has
   reached `CHUNK_MIN_SIZE`; force a break at `CHUNK_MAX_SIZE`; otherwise close
   at `CHUNK_TARGET_SIZE`.
7. Fold under-sized chunks back into a neighbour when it does not overflow.
8. Extend each chunk backwards by `CHUNK_OVERLAP` sentences, bounded by max size.
9. Drop chunks whose text already appeared in the same document.

### Tuning

Defaults were set from measured section sizes (p25 = 92, median = 157,
p75 = 430, p95 = 2 108 chars — only 6.4% of sections exceed 1 600), not picked
arbitrarily:

```env
CHUNK_TARGET_SIZE=700
CHUNK_MIN_SIZE=200
CHUNK_MAX_SIZE=1400
SEMANTIC_SIMILARITY_THRESHOLD=0.55
CHUNK_OVERLAP=1        # in sentences
```

## 8. Sentence segmentation

Deterministic, single pass, no model. Handles Arabic (`؟`, `۔`), English and
mixed text. Before splitting it masks the constructs that cause false breaks in
this corpus: URLs, emails, **USSD codes** (`*880*1#`), decimals and prices
(`1.5 GB`, `70.50 EGP`), ellipses and list ordinals. Abbreviations (`Dr.`,
`e.g.`) do not split. Units and currencies are deliberately *not* treated as
abbreviations, because `costs 70.50 EGP.` legitimately ends a sentence. Newlines
are hard boundaries so list items and table rows stay separate.

## 9. Embedding model

`EmbeddingService` exposes `embed_documents()`, `embed_query()` and
`model_info()`. Nothing else in the codebase knows which vendor is in use.

| provider | model | use |
|---|---|---|
| `openai` *(default)* | `text-embedding-3-large` | strong multilingual incl. Arabic |
| `cohere` | `embed-multilingual-v3.0` | trained explicitly for Arabic + English; uses `search_query`/`search_document` input types |
| `sentence_transformers` | `intfloat/multilingual-e5-base` | optional, fully offline (needs torch) |
| `local_lexical` | hashed char-n-gram | **dev/test only** — real but lexical similarity |
| `hash` | deterministic | unit tests only |

A `CachedEmbedding` wrapper memoises by content hash, which matters twice: the
chunker embeds every sentence, and re-ingesting unchanged text would otherwise
repay the full API cost.

## 10. Chroma Cloud

Chroma Cloud is the **only** vector database. There is no `PersistentClient`
path and nothing is written to disk. Credentials come from the environment and
are never logged; `store info` prints `<set>`/`<unset>`, never the key.

`VectorStore` exposes `add_chunks`, `upsert_chunks`, `delete_chunks`,
`delete_document`, `similarity_search`, `count`, `info`. The client is built via
`chromadb.CloudClient` when available, falling back to `HttpClient` against
`api.trychroma.com` for older pinned versions. Writes are batched (100/request)
and transient failures retry with exponential backoff.

### Idempotent ingestion

Chunk ids are deterministic:

```text
{document_id}:{section_index}:{chunk_index}:{hash(section_path, text)[:12]}
```

`document_id` is a stable hash of `source_url`. Ingestion is
**delete-then-upsert per document**: any stored chunk whose id is not in the
freshly computed set is pruned, then the new set is upserted. Re-running on
unchanged input produces byte-identical ids and cannot create duplicates —
verified: two consecutive full runs left exactly 9 333 vectors.

Metadata stored per chunk is compact and never includes raw HTML: `chunk_id`,
`document_id`, `source_url`, `section_title`, `section_path` (joined with ` > `,
since Chroma takes scalars only), `heading_level`, `section_index`,
`chunk_index`, `chunking_method`, `section_extraction_method`, `language`,
`domain`, `scraped_at`, `dataset_file`, `component_type`, `sentence_count`,
`content_hash`.

## 11. Retrieval

`RetrievalService.retrieve(query, top_k, filters)` runs independently of
generation. It returns text, score (`1 − cosine distance`), distance, source
URL, section title, section path and full metadata, and records embedding vs
vector-search latency separately.

Filters are restricted to an explicit allow-list — `domain`, `language`,
`document_id`, `source_url`, `dataset_file` — so a caller cannot craft an
arbitrary Chroma `where` clause. Unknown keys are dropped with a warning.

Because the corpus publishes identical content under several URLs, retrieval
over-fetches (`top_k × 3`) and collapses passages with the same content hash,
keeping the best-scoring copy and listing the others in
`metadata.duplicate_source_urls`. Disable with `DEDUPLICATE_RESULTS=false`.

## 12. Generation and citations

`GenerationService` builds a numbered context block — each passage labelled
`[n]` with its section path and URL — and prompts the model to answer **only**
from it, in the user's language, citing by number. If the context is
insufficient the model returns an `INSUFFICIENT_CONTEXT` marker and the service
returns an explicit "not enough information" answer with zero citations.

**The model never writes a URL.** It emits `[1]`, `[2]`; the citation builder
maps those indices back onto retrieved metadata and discards any index that does
not exist. A fabricated `[9]` is stripped from the answer and produces no
citation — covered by a test.

## 13. Evaluation methodology

Evaluation is split by pipeline stage so a bad end-to-end score can be
attributed:

| stage | command | needs |
|---|---|---|
| 1. Chunking | `eval chunking` | nothing (no store, no LLM) |
| 2. Embedding / retrieval | `eval retrieval` | vector store |
| 3-6. Context, answer, citations, end-to-end | `eval rag` | vector store + LLM |

Deterministic metrics and LLM-as-a-judge metrics are reported under separate
keys so they are never confused. **A metric whose ground truth is missing is
reported as `null` with a reason under `unavailable_metrics` — never as 0.**
If generation runs on the offline stub, every answer-side metric is marked
unavailable rather than presented as quality.

Ragas is deliberately not used: it would pull in LangChain for metrics this
project already computes directly, and production retrieval must not depend on
an evaluation library.

## 14. Evaluation metrics

**Retrieval (deterministic):** HitRate@{1,3,5,10}, Recall@{1,3,5,10},
Precision@{1,3,5,10} (denominator `k`, so short result lists are penalised),
MRR, nDCG@{3,5,10} with graded relevance when the dataset provides it, section
retrieval accuracy, source retrieval accuracy, and latency (mean / median / p95
/ p99, with embedding time separated from search time).

**Answer quality (deterministic):** citation validity, citation correctness
(lexical), citation completeness (share of factual sentences that carry a
citation), lexical groundedness, answer token-F1, insufficient-context handling.

**Answer quality (LLM judge, reported separately):** faithfulness, answer
relevance, context relevance, context precision, context recall, answer
correctness, unsupported claim rate.

**Chunking:** chunk count, size mean/median/p25/p75/p95, sentences per chunk,
chunks per document/section, % at max size, % below min size, section-boundary
violations, empty chunks, duplicate chunks. `--compare` runs two configurations
side by side.

## 15. Future LangGraph extension point

Current workflow — linear, so plain Python is the right tool:

```text
Retrieve
   ↓
Generate
```

Possible future workflow:

```text
Retrieve
   ↓
Evaluate Context
   ↓
Relevant?
 ├── YES → Generate
 └── NO  → Rewrite Query
              ↓
           Retrieve Again
```

`services/interfaces.py` already defines `Retriever`, `ContextEvaluator`,
`QueryRewriter` and `Generator` as protocols. `Retriever` and `Generator` are
implemented; `ContextEvaluator` and `QueryRewriter` are intentionally left
unimplemented — a stub evaluator that always returned "relevant" would be worse
than none.

LangGraph becomes appropriate when the workflow acquires **state, branching and
loops**: retry budgets, per-attempt query history, conditional edges,
checkpointing, human-in-the-loop review. None of those exist today, so adding a
graph runtime now would be pure overhead. Because the services are already
independent, introducing it later is a wiring change, not a rewrite.

## 16. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e rag/            # or: pip install -r rag/requirements.txt
cp rag/.env.example rag/.env   # then fill in credentials
```

Optional, for fully offline embeddings (pulls in torch, ~2 GB):

```bash
pip install -r rag/requirements-local-embeddings.txt
```

## 17. Environment variables

See `.env.example`. `.env` is git-ignored and must never be committed.

```env
CHROMA_API_KEY=            # never logged
CHROMA_TENANT=
CHROMA_DATABASE=
CHROMA_COLLECTION_NAME=vf_egypt_kb

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=
COHERE_API_KEY=

LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=

RETRIEVAL_TEXT_POLICY=enhanced_first   # | language_aware | raw_only

CHUNK_TARGET_SIZE=700
CHUNK_MIN_SIZE=200
CHUNK_MAX_SIZE=1400
SEMANTIC_SIMILARITY_THRESHOLD=0.55
CHUNK_OVERLAP=1

TOP_K=5
DEDUPLICATE_RESULTS=true
```

## 18. CLI commands

```bash
# Dataset
rag-cli dataset inspect --limit 2900 --json rag/data/evaluation/dataset_report.json
rag-cli dataset extraction --limit 800

# Ingestion (idempotent)
rag-cli ingest run                          # whole rag/data/
rag-cli ingest run --file rag/data/batch_0000.parquet
rag-cli ingest run --directory rag/data --limit 1200
rag-cli ingest run --dry-run                # chunk without writing vectors
rag-cli ingest delete --document-id <id>    # clean re-ingest

# Inspection
rag-cli inspect sections --url https://web.vodafone.com.eg/en/vodafone-flex
rag-cli inspect chunks   --url https://web.vodafone.com.eg/en/vodafone-flex

# Retrieval and full RAG
rag-cli retrieve --query "ازاي أجدد باقة الإنترنت؟" --top-k 5
rag-cli retrieve --query "..." --domain web.vodafone.com.eg --language ar --json
rag-cli ask      --query "How do I renew my Flex bundle?" --top-k 5

# Vector store
rag-cli store info      # collection, vector count, connection; never the key
rag-cli store config    # effective settings, secrets redacted

# Evaluation
rag-cli eval build-dataset
rag-cli eval chunking --limit 600 --compare
rag-cli eval retrieval --top-k 10
rag-cli eval rag --top-k 5 --judge
rag-cli eval summary

# Tests
pytest rag/tests -q
```

Add `--offline --offline-store /tmp/store.json` to any command to run against an
in-memory store instead of Chroma Cloud (useful before credentials exist). The
snapshot is JSON, not a Chroma database, and is written outside `rag/data/`.

`--embedding-provider` / `--embedding-model` override the configured model on
any command, which is how two models are compared on the same questions.

## 19. API endpoints

```bash
uvicorn rag.api.app:app --reload
# docs at http://127.0.0.1:8000/docs
```

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness + vector store / embedding diagnostics |
| POST | `/api/v1/rag/retrieve` | retrieval only, no generation |
| POST | `/api/v1/rag/query` | full grounded answer with citations |

```bash
curl -s localhost:8000/api/v1/rag/retrieve -H 'Content-Type: application/json' -d '{
  "query": "ازاي أجدد باقة الإنترنت؟",
  "top_k": 5,
  "filters": {"language": "ar"}
}'

curl -s localhost:8000/api/v1/rag/query -H 'Content-Type: application/json' -d '{
  "query": "How do I renew my Flex bundle?",
  "top_k": 5,
  "include_chunks": true
}'
```

To mount inside the parent project instead of running standalone:

```python
from rag.api.app import rag_router
app.include_router(rag_router, prefix="/api/v1")
```

## 20. Testing

```bash
pytest rag/tests -q      # 106 tests
```

Covers heading extraction and nested hierarchy, section paths, accordion/FAQ/tab
detection, navigation accordions and plain buttons *not* becoming sections,
malformed HTML, Arabic/English/mixed sentence splitting, USSD and decimal
protection, semantic boundaries, min/max/target sizing, section-boundary
preservation, deterministic chunk ids, Chroma insert/upsert/duplicate
prevention/delete/re-ingest/metadata/filters (against a mocked client), the
three API endpoints, fabricated-citation stripping, and every metric against
hand-computed examples (including the standard nDCG worked example).

No test requires live Chroma Cloud credentials or a real LLM: the Chroma client
is mocked, and generation and judging use stubs.

## 21. Known limitations

- **`llm_enhanced_text` is English for 88% of Arabic pages.** With
  `RETRIEVAL_TEXT_POLICY=enhanced_first` (the configured default), documents
  that fall back to text-based extraction will carry English chunks for Arabic
  pages. Set `language_aware` to avoid this.
- **Cross-document duplication is high.** ~55% of chunk instances share text
  with another chunk, because the site publishes the same content at several
  URLs. Mitigated at retrieval time, not at index time, so no URL is lost.
- **Heading hierarchy is only as good as the markup.** Where a page puts an
  `h3` after an unrelated `h2`, the `section_path` inherits that `h2`. This is
  inherent to flat HTML heading structure.
- **Section retrieval accuracy is measured by exact section-path match**, so a
  correct answer retrieved under a differently-worded path scores as a miss.
- **Log redaction is pattern-based defence in depth.** It catches `sk-`/`ck-`
  prefixed keys anywhere and any value in a key-labelled context
  (`CHROMA_API_KEY=…`, `Bearer …`, `x-chroma-token=…`), but a bare opaque
  string with no surrounding label cannot be distinguished from ordinary text.
  The primary guarantee is that no code path formats a credential into a log
  message; redaction is the backstop.
- The evaluation set is 51 questions. It is large enough to compare
  configurations, not to certify production quality.
