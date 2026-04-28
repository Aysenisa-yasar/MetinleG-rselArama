"""Visualization helpers for qualitative and quantitative retrieval outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def _open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def plot_top_k_results(query: str, results_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a figure showing a query and its top-k retrieved images."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_results = len(results_df)
    fig, axes = plt.subplots(1, num_results, figsize=(4 * num_results, 4))
    if num_results == 1:
        axes = [axes]

    for axis, (_, row) in zip(axes, results_df.iterrows()):
        axis.imshow(_open_image(row["image_path"]))
        axis.set_title(f"#{row['rank']} | {row['score']:.3f}")
        axis.set_axis_off()

        captions = row["captions"]
        if isinstance(captions, list) and captions:
            axis.text(
                0.5,
                -0.15,
                captions[0][:70],
                ha="center",
                va="top",
                transform=axis.transAxes,
                fontsize=9,
                wrap=True,
            )

    fig.suptitle(f"Sorgu: {query}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recall_bar_chart(metrics: dict[str, float], output_path: str | Path) -> None:
    """Save a bar chart for Recall@K and MRR."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [name for name in metrics if name.startswith("Recall@")] + ["MRR"]
    metric_values = [metrics[name] for name in metric_names]

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(metric_names, metric_values, color=["#005f73", "#0a9396", "#94d2bd", "#ee9b00"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Retrieval Metrics")
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    for idx, value in enumerate(metric_values):
        axis.text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_cases(
    cases_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    max_examples: int = 4,
) -> None:
    """Save a retrieved-vs-ground-truth comparison figure for selected cases."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = cases_df.head(max_examples).copy()
    if subset.empty:
        return

    num_rows = len(subset)
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))
    if num_rows == 1:
        axes = [axes]

    for row_index, (_, row) in enumerate(subset.iterrows()):
        left_axis, right_axis = axes[row_index]
        left_axis.imshow(_open_image(row["top1_image_path"]))
        left_axis.set_title(f"Retrieved | rank={row['rank']} | score={row['top1_score']:.3f}")
        left_axis.set_axis_off()

        right_axis.imshow(_open_image(row["ground_truth_image_path"]))
        right_axis.set_title("Ground truth")
        right_axis.set_axis_off()

        left_axis.text(
            0.0,
            -0.15,
            f"Sorgu: {row['query'][:80]}",
            transform=left_axis.transAxes,
            fontsize=10,
            va="top",
            wrap=True,
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rank_distribution(
    detailed_df: pd.DataFrame,
    output_path: str | Path,
    max_rank: int = 50,
) -> None:
    """Save a histogram showing where the correct image appears in the ranking."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if detailed_df.empty:
        return

    ranks = detailed_df["rank"].clip(upper=max_rank)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(ranks, bins=min(max_rank, 25), color="#ae2012", alpha=0.85)
    axis.set_xlabel(f"Rank (>{max_rank} clipped to {max_rank})")
    axis.set_ylabel("Query count")
    axis.set_title("Ground Truth Rank Distribution")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
