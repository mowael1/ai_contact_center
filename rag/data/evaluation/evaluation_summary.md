# RAG evaluation summary

- Collection: `vf_egypt_kb`
- Embedding model: `text-embedding-3-large`
- LLM model: `claude-sonnet-5`
- Chunking: {'target': 700, 'min': 200, 'max': 1400, 'similarity_threshold': 0.55, 'overlap_sentences': 1}

## Retrieval
- Questions evaluated: 48
- HitRate@1: 0.4375
- HitRate@10: 0.8125
- HitRate@3: 0.6458
- HitRate@5: 0.75
- MRR: 0.5622
- Precision@1: 0.4375
- Precision@10: 0.2563
- Precision@3: 0.3611
- Precision@5: 0.3542
- Recall@1: 0.4375
- Recall@10: 0.8125
- Recall@3: 0.6458
- Recall@5: 0.75
- nDCG@10: 0.6141
- nDCG@3: 0.4574
- nDCG@5: 0.5414
- Section retrieval accuracy: {'top_1': 0.1, 'top_3': 0.3, 'top_5': 0.475, 'top_10': 0.5}
- Source retrieval accuracy: {'top_1': 0.4375, 'top_3': 0.6458, 'top_5': 0.75, 'top_10': 0.8125}
- Latency ms: mean=635.859 median=635.031 p95=655.817 p99=673.476

## RAG answer quality
- Questions: 51
### Deterministic
- answer_token_f1: 0.0448
- citation_correctness_lexical: 0.2725
- insufficient_context_handling: 0.9412
- lexical_groundedness: 0.1106
### Unavailable
- answer_correctness: LLM judge disabled (no LLM credentials or --no-judge)
- answer_relevance: LLM judge disabled (no LLM credentials or --no-judge)
- answer_token_f1: generation ran on the offline stub (LLM_PROVIDER=echo); set a real LLM_PROVIDER and API key to measure this
- citation_completeness: generation ran on the offline stub (LLM_PROVIDER=echo); set a real LLM_PROVIDER and API key to measure this
- citation_correctness_lexical: generation ran on the offline stub (LLM_PROVIDER=echo); set a real LLM_PROVIDER and API key to measure this
- citation_validity: generation ran on the offline stub (LLM_PROVIDER=echo); set a real LLM_PROVIDER and API key to measure this
- context_precision: LLM judge disabled (no LLM credentials or --no-judge)
- context_recall: LLM judge disabled (no LLM credentials or --no-judge)
- context_relevance: LLM judge disabled (no LLM credentials or --no-judge)
- faithfulness: LLM judge disabled (no LLM credentials or --no-judge)
- insufficient_context_handling: generation ran on the offline stub (LLM_PROVIDER=echo); set a real LLM_PROVIDER and API key to measure this
- lexical_groundedness: generation ran on the offline stub (LLM_PROVIDER=echo); set a real LLM_PROVIDER and API key to measure this
- unsupported_claim_rate: LLM judge disabled (no LLM credentials or --no-judge)
### LLM usage
- input=0 output=0 total=0 cost_usd=0.0
