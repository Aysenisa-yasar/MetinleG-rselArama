"""Evaluation metrics for retrieval tasks."""

from __future__ import annotations

import pandas as pd


def recall_at_k(ranks: list[int], k: int) -> float:
    """Compute Recall@k from 1-indexed ranks."""
    if not ranks:
        return 0.0
    return sum(rank <= k for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """Compute the mean reciprocal rank from 1-indexed ranks."""
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def summarize_metrics(ranks: list[int], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Create a metric summary dictionary."""
    summary = {f"Recall@{k}": recall_at_k(ranks, k) for k in ks}
    summary["MRR"] = mean_reciprocal_rank(ranks)
    summary["MedianRank"] = float(pd.Series(ranks).median()) if ranks else 0.0
    summary["MeanRank"] = float(pd.Series(ranks).mean()) if ranks else 0.0
    return summary


def metrics_to_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    """Convert metric dictionary to a tidy dataframe."""
    return pd.DataFrame(
        [{"metric": metric_name, "value": metric_value} for metric_name, metric_value in metrics.items()]
    )

