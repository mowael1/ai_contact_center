"""Deterministic retrieval metrics.

Every function here is pure and closed-form - no model, no judge - so the
numbers are reproducible. Graded relevance is supported where the evaluation
dataset provides it; otherwise binary relevance is assumed.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


def hit_rate_at_k(relevance: Sequence[float], k: int) -> float:
    """1.0 if any of the top-k results is relevant."""
    return 1.0 if any(r > 0 for r in relevance[:k]) else 0.0


def precision_at_k(relevance: Sequence[float], k: int) -> float:
    """Share of the top-k results that are relevant.

    The denominator is ``k`` (not the number returned) so that returning fewer
    than k results is correctly penalised.
    """
    if k <= 0:
        return 0.0
    return sum(1 for r in relevance[:k] if r > 0) / k


def recall_at_k(relevance: Sequence[float], k: int, total_relevant: int) -> float:
    """Share of all known-relevant items that appear in the top-k."""
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for r in relevance[:k] if r > 0)
    return min(1.0, found / total_relevant)


def reciprocal_rank(relevance: Sequence[float]) -> float:
    """1 / rank of the first relevant result, else 0."""
    for i, r in enumerate(relevance, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(gains: Sequence[float], k: int, ideal_gains: Sequence[float] | None = None) -> float:
    """Normalised DCG. Uses graded gains when the dataset supplies them."""
    ideal = sorted(ideal_gains if ideal_gains is not None else gains, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; returns 0.0 for an empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return float(ordered[index])


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    if not latencies:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "count": len(latencies),
        "mean_ms": round(statistics.fmean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "p99_ms": round(percentile(latencies, 99), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
    }


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across queries, skipping metrics with no data."""
    if not per_query:
        return {}
    keys: list[str] = []
    for row in per_query:
        for key in row:
            if key not in keys:
                keys.append(key)
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in per_query if row.get(key) is not None]
        if values:
            out[key] = round(statistics.fmean(values), 4)
    return out
