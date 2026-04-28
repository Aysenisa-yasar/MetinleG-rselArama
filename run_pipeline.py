"""Command-line entrypoint for the Turkish image retrieval project."""

from __future__ import annotations

import argparse

from src.config import ProjectConfig, get_logger, set_seed
from src.data_loader import prepare_data
from src.embedder import embed_all_splits
from src.evaluate import evaluate_split
from src.train import train_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turkce metin-gorsel retrieval pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare_data", "embed", "evaluate", "train", "all"],
        help="Pipeline stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    config.ensure_directories()
    set_seed(config.seed)
    logger = get_logger("pipeline", config.log_path())

    if args.stage in {"prepare_data", "all"}:
        logger.info("Stage: prepare_data")
        prepare_data(config)

    if args.stage in {"embed", "all"}:
        logger.info("Stage: embed")
        embed_all_splits(config)

    if args.stage in {"evaluate", "all"}:
        logger.info("Stage: evaluate")
        metrics = evaluate_split(config, split="test", top_k=config.max_eval_top_k)
        logger.info("Evaluation metrics: %s", metrics)

    if args.stage == "train":
        logger.info("Stage: train")
        outputs = train_text_encoder(config)
        logger.info("Best validation selection score: %.4f", outputs.best_score)
        logger.info("Best validation metrics: %s", outputs.best_val_metrics)
        logger.info("Fine-tuned test metrics: %s", outputs.test_metrics)


if __name__ == "__main__":
    main()
