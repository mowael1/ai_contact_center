# Embedding model comparison — EmbeddingGemma vs BGE-M3

Controlled A/B on an identical chunk pool. The **only** variable is the embedding model.

- Pool: **400 chunks** (243 from documents holding a ground-truth answer, 157 distractors, seeded shuffle)
- Questions: **48** (Arabic 23, English 23), top_k=10
- Chunks, questions and ground truth are byte-identical between the two runs.

## Overall

| metric | EmbeddingGemma-300m | BGE-M3 | delta |
|---|---|---|---|
| HitRate@1 | 0.6667 | 0.7917 | +0.1250 |
| HitRate@3 | 0.8750 | 0.9167 | +0.0417 |
| HitRate@5 | 0.9583 | 0.9583 | =+0.0000 |
| HitRate@10 | 0.9792 | 0.9792 | =+0.0000 |
| Precision@1 | 0.6667 | 0.7917 | +0.1250 |
| Precision@5 | 0.4750 | 0.5792 | +0.1042 |
| MRR | 0.7776 | 0.8634 | +0.0858 |
| nDCG@10 | 0.7671 | 0.8506 | +0.0835 |

## Arabic questions only

| metric | EmbeddingGemma-300m | BGE-M3 | delta |
|---|---|---|---|
| HitRate@1 | 0.7391 | 0.7826 | +0.0435 |
| HitRate@5 | 1.0000 | 0.9565 | -0.0435 |
| HitRate@10 | 1.0000 | 0.9565 | -0.0435 |
| MRR | 0.8348 | 0.8587 | +0.0239 |
| nDCG@10 | 0.8303 | 0.8424 | +0.0121 |

## English questions only

| metric | EmbeddingGemma-300m | BGE-M3 | delta |
|---|---|---|---|
| HitRate@1 | 0.6522 | 0.8261 | +0.1739 |
| HitRate@5 | 0.9130 | 1.0000 | +0.0870 |
| HitRate@10 | 0.9565 | 1.0000 | +0.0435 |
| MRR | 0.7446 | 0.8949 | +0.1503 |
| nDCG@10 | 0.7183 | 0.8797 | +0.1614 |

## Cost / operational

| | EmbeddingGemma-300m | BGE-M3 |
|---|---|---|
| dimensions | 768 | 1024 |
| index 400 chunks | 36.5s | 67.4s |
| mean query latency | 324ms | 1658ms |
| extrapolated full-corpus ingest (19,125 chunks) | ~29 min | ~54 min |
| vector storage vs 768d | 1.0x | 1.33x |

## Caveat

A 400-chunk pool is far easier than the production index of 19,125 chunks — both models score higher here than they would in production. The comparison between them is valid; the absolute numbers are optimistic.
