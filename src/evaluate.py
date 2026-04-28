"""Retrieval evaluation for Turkish caption queries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.config import ProjectConfig, get_logger
from src.data_loader import load_processed_split, prepare_data
from src.embedder import (
    EmbeddingStore,
    MultilingualClipEmbedder,
    embed_all_splits,
    embed_split,
    load_embedding_store,
)
from src.metrics import metrics_to_dataframe, summarize_metrics
from src.preprocess import (
    flatten_caption_records,
    slugify_text,
    token_overlap,
    tokenize_for_analysis,
    top_informative_tokens,
)
from src.retrieval import build_retrieval_index, retrieve_images
from src.visualize import (
    plot_comparison_cases,
    plot_rank_distribution,
    plot_recall_bar_chart,
    plot_top_k_results,
)


def _ensure_processed_data(config: ProjectConfig) -> None:
    """Prepare processed records if they are missing."""
    if not config.processed_records_path("test").exists():
        prepare_data(config)


def _ensure_embedding_store(
    config: ProjectConfig,
    split: str,
    tag: str | None = None,
    text_model_name_or_path: str | Path | None = None,
    image_model_name_or_path: str | Path | None = None,
) -> EmbeddingStore:
    """Load split embeddings, generating them if necessary."""
    embedding_path = config.split_embedding_path(split, tag=tag)
    metadata_path = config.split_embedding_metadata_path(split, tag=tag)
    if embedding_path.exists() and metadata_path.exists():
        return load_embedding_store(split, config, tag=tag)

    if tag is None or tag == config.baseline_tag:
        embed_all_splits(config)
    else:
        embedder = MultilingualClipEmbedder(
            config=config,
            text_model_name_or_path=text_model_name_or_path,
            image_model_name_or_path=image_model_name_or_path,
        )
        embed_split(split=split, embedder=embedder, config=config, tag=tag)

    return load_embedding_store(split, config, tag=tag)


def _rank_bucket(rank: int) -> str:
    """Bucketize rank values for easier failure analysis."""
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 50:
        return "11-50"
    return "51+"


def _average_recall(metrics: dict[str, float]) -> float:
    """Aggregate Recall@1/5/10 into a single checkpoint-selection score."""
    return float((metrics["Recall@1"] + metrics["Recall@5"] + metrics["Recall@10"]) / 3.0)


def compute_retrieval_details(
    queries_df: pd.DataFrame,
    image_metadata: pd.DataFrame,
    image_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute detailed per-query retrieval outputs and summary metrics."""
    scores = torch.matmul(query_embeddings, image_embeddings.T)
    sorted_indices = torch.argsort(scores, dim=1, descending=True)

    image_id_to_index = {
        str(image_id): index for index, image_id in enumerate(image_metadata["image_id"].tolist())
    }
    target_indices = torch.tensor(
        [image_id_to_index[str(image_id)] for image_id in queries_df["image_id"].tolist()],
        dtype=torch.long,
    )

    matches = sorted_indices.eq(target_indices.unsqueeze(1))
    ranks = (matches.float().argmax(dim=1) + 1).tolist()
    metrics = summarize_metrics(ranks, ks=(1, 5, 10))
    metrics["AvgRecall"] = _average_recall(metrics)

    effective_top_k = min(top_k, scores.shape[1])
    top_scores, top_indices = torch.topk(scores, k=effective_top_k, dim=1)
    ground_truth_scores = scores.gather(1, target_indices.unsqueeze(1)).squeeze(1)

    detailed_rows: list[dict] = []
    for row_idx, rank in enumerate(ranks):
        top_candidate_indices = top_indices[row_idx].tolist()
        top_candidate_scores = top_scores[row_idx].tolist()
        top1_index = top_candidate_indices[0]
        top1_row = image_metadata.iloc[top1_index]
        ground_truth_row = image_metadata.iloc[target_indices[row_idx].item()]
        query_text = queries_df.iloc[row_idx]["caption"]
        top1_primary_caption = top1_row["captions"][0] if top1_row["captions"] else ""
        gt_primary_caption = ground_truth_row["captions"][0] if ground_truth_row["captions"] else ""

        detailed_rows.append(
            {
                "query": query_text,
                "query_token_count": len(tokenize_for_analysis(query_text)),
                "ground_truth_image_id": queries_df.iloc[row_idx]["image_id"],
                "ground_truth_image_path": ground_truth_row["image_path"],
                "ground_truth_primary_caption": gt_primary_caption,
                "ground_truth_captions": json.dumps(ground_truth_row["captions"], ensure_ascii=False),
                "ground_truth_score": float(ground_truth_scores[row_idx].item()),
                "rank": int(rank),
                "rank_bucket": _rank_bucket(int(rank)),
                "top1_image_id": top1_row["image_id"],
                "top1_image_path": top1_row["image_path"],
                "top1_primary_caption": top1_primary_caption,
                "top1_captions": json.dumps(top1_row["captions"], ensure_ascii=False),
                "top1_score": float(top_candidate_scores[0]),
                "score_gap_top1_vs_gt": float(top_candidate_scores[0] - ground_truth_scores[row_idx].item()),
                "top1_caption_token_overlap": token_overlap(query_text, top1_primary_caption),
                "ground_truth_caption_token_overlap": token_overlap(query_text, gt_primary_caption),
                "topk_image_ids": json.dumps(
                    [str(image_metadata.iloc[idx]["image_id"]) for idx in top_candidate_indices],
                    ensure_ascii=False,
                ),
                "topk_scores": json.dumps([round(float(score), 6) for score in top_candidate_scores]),
            }
        )

    return pd.DataFrame(detailed_rows), metrics


def _write_error_summary(
    detailed_df: pd.DataFrame,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Write a compact plain-text error analysis report."""
    failed_df = detailed_df[detailed_df["rank"] > 1].copy()
    success_df = detailed_df[detailed_df["rank"] == 1].copy()

    lines = [
        "Retrieval Error Analysis",
        "",
        f"Total queries: {len(detailed_df)}",
        f"Successful @1: {len(success_df)}",
        f"Failures: {len(failed_df)}",
        "",
    ]
    lines.extend(f"{metric}: {value:.4f}" for metric, value in metrics.items())
    lines.append("")

    if failed_df.empty:
        lines.append("No failed queries were found for this evaluation run.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    rank_bucket_counts = failed_df["rank_bucket"].value_counts().sort_index()
    lines.append("Failure buckets:")
    lines.extend(f"- {bucket}: {count}" for bucket, count in rank_bucket_counts.items())
    lines.append("")

    lines.append(
        "Average query token count | success={:.2f} | failure={:.2f}".format(
            success_df["query_token_count"].mean() if not success_df.empty else 0.0,
            failed_df["query_token_count"].mean(),
        )
    )
    lines.append("")

    failed_query_tokens = top_informative_tokens(failed_df["query"].tolist(), top_n=10)
    mistaken_caption_tokens = top_informative_tokens(failed_df["top1_primary_caption"].tolist(), top_n=10)
    lines.append("Common tokens in failed queries:")
    lines.extend(f"- {token}: {count}" for token, count in failed_query_tokens)
    lines.append("")
    lines.append("Common tokens in mistaken top-1 captions:")
    lines.extend(f"- {token}: {count}" for token, count in mistaken_caption_tokens)
    lines.append("")

    hardest_failures = failed_df.sort_values(["rank", "score_gap_top1_vs_gt"], ascending=[False, False]).head(10)
    lines.append("Hardest failures:")
    for _, row in hardest_failures.iterrows():
        lines.append(
            "- rank={rank} | query={query} | predicted={pred} | ground_truth={gt}".format(
                rank=int(row["rank"]),
                query=row["query"][:90],
                pred=row["top1_primary_caption"][:90],
                gt=row["ground_truth_primary_caption"][:90],
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_split(
    config: ProjectConfig,
    split: str = "test",
    top_k: int = 10,
    text_model_name_or_path: str | Path | None = None,
    image_model_name_or_path: str | Path | None = None,
    tag: str | None = None,
    backend: str | None = None,
) -> dict[str, float]:
    """Evaluate retrieval by using every caption as a query against its split images."""
    logger = get_logger("evaluate", config.log_path())
    config.ensure_directories()
    _ensure_processed_data(config)

    embedding_store = _ensure_embedding_store(
        config=config,
        split=split,
        tag=tag,
        text_model_name_or_path=text_model_name_or_path,
        image_model_name_or_path=image_model_name_or_path,
    )
    image_embeddings = embedding_store.embeddings
    image_metadata = embedding_store.metadata
    metadata_df = load_processed_split(split, config)
    queries_df = flatten_caption_records(metadata_df)

    logger.info(
        "Evaluating %s caption queries against %s images | split=%s | tag=%s",
        len(queries_df),
        len(image_metadata),
        split,
        tag or config.baseline_tag,
    )
    embedder = MultilingualClipEmbedder(
        config=config,
        text_model_name_or_path=text_model_name_or_path,
        image_model_name_or_path=image_model_name_or_path,
    )
    query_embeddings = embedder.encode_texts(queries_df["caption"].tolist())
    detailed_df, metrics = compute_retrieval_details(
        queries_df=queries_df,
        image_metadata=image_metadata,
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        top_k=top_k,
    )

    detailed_df.to_csv(config.detailed_results_path(split, tag=tag), index=False)
    failed_df = detailed_df[detailed_df["rank"] > 1].copy()
    failed_df.to_csv(config.error_analysis_path(split, tag=tag), index=False)

    metrics_df = metrics_to_dataframe(metrics)
    metrics_df.to_csv(config.metrics_csv_path(split, tag=tag), index=False)

    summary_lines = [
        f"Split: {split}",
        f"Tag: {tag or config.baseline_tag}",
        f"Queries evaluated: {len(queries_df)}",
        f"Images searched: {len(image_metadata)}",
        f"Backend: {backend or config.retrieval_backend}",
        "",
    ]
    summary_lines.extend(f"{metric}: {value:.4f}" for metric, value in metrics.items())
    config.summary_txt_path(split, tag=tag).write_text("\n".join(summary_lines), encoding="utf-8")
    _write_error_summary(detailed_df, metrics, config.error_summary_path(split, tag=tag))

    plot_recall_bar_chart(metrics, config.figure_path(f"{split}_recall_at_k.png", tag=tag))
    plot_rank_distribution(detailed_df, config.figure_path(f"{split}_rank_distribution.png", tag=tag))

    successful_cases = detailed_df[detailed_df["rank"] == 1].head(4)
    failed_cases = detailed_df[detailed_df["rank"] > 1].sort_values("rank", ascending=False).head(4)
    plot_comparison_cases(
        successful_cases,
        config.figure_path(f"{split}_successful_examples.png", tag=tag),
        title="Successful Retrieval Examples",
    )
    plot_comparison_cases(
        failed_cases,
        config.figure_path(f"{split}_failed_examples.png", tag=tag),
        title="Failed Retrieval Examples",
    )

    retrieval_index = build_retrieval_index(split, config, tag=tag, backend=backend)
    sample_query_rows: list[dict] = []
    for query in config.sample_queries:
        results_df = retrieve_images(
            query=query,
            index=retrieval_index,
            embedder=embedder,
            top_k=config.top_k_default,
        )
        plot_top_k_results(
            query=query,
            results_df=results_df,
            output_path=config.figure_path(
                f"{split}_{slugify_text(query)}_top{config.top_k_default}.png",
                tag=tag,
            ),
        )

        for row in results_df.to_dict(orient="records"):
            sample_query_rows.append(
                {
                    "query": query,
                    "rank": row["rank"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "score": row["score"],
                    "captions": json.dumps(row["captions"], ensure_ascii=False),
                }
            )

    pd.DataFrame(sample_query_rows).to_csv(config.sample_queries_path(split, tag=tag), index=False)
    logger.info("Evaluation reports saved to %s", config.reports_dir)
    return metrics
