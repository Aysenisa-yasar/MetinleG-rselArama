"""Project configuration, path management, and shared utilities."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may not exist before install
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ProjectConfig:
    """Holds project-wide paths and runtime configuration."""

    dataset_name: str = "atasoglu/flickr8k-turkish"
    text_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    image_model_name: str = "clip-ViT-B-32"
    baseline_tag: str = "baseline"
    finetuned_tag: str = "finetuned_best"
    seed: int = 42
    image_batch_size: int = 32
    text_batch_size: int = 64
    train_batch_size: int = 16
    num_workers: int = 0
    num_epochs: int = 2
    learning_rate: float = 2e-5
    text_learning_rate: float = 1e-5
    vision_learning_rate: float = 5e-6
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    temperature: float = 0.07
    text_unfreeze_layers: int = 2
    vision_unfreeze_layers: int = 2
    hard_negative_pool_size: int = 8
    hard_negative_count: int = 1
    hard_negative_weight: float = 0.35
    hard_negative_margin: float = 0.08
    checkpoint_selection_metric: str = "avg_recall"
    retrieval_backend: str = "exact"
    lsh_num_tables: int = 8
    lsh_num_planes: int = 12
    lsh_min_candidates: int = 128
    top_k_default: int = 5
    max_eval_top_k: int = 10
    image_format: str = "jpg"
    report_encoding: str = "utf-8"

    root_dir: Path = ROOT_DIR
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / "data")
    raw_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "processed")
    embeddings_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "embeddings")
    outputs_dir: Path = field(default_factory=lambda: ROOT_DIR / "outputs")
    figures_dir: Path = field(default_factory=lambda: ROOT_DIR / "outputs" / "figures")
    logs_dir: Path = field(default_factory=lambda: ROOT_DIR / "outputs" / "logs")
    reports_dir: Path = field(default_factory=lambda: ROOT_DIR / "outputs" / "reports")
    checkpoints_dir: Path = field(default_factory=lambda: ROOT_DIR / "outputs" / "checkpoints")
    model_cache_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "hf_cache")

    sample_queries: list[str] = field(
        default_factory=lambda: [
            "deniz kenarinda kosan kopek",
            "parkta oynayan cocuk",
            "karda duran insanlar",
            "kanoda kurek ceken kadin",
        ]
    )

    def ensure_directories(self) -> None:
        """Create all required project directories if they are missing."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.embeddings_dir,
            self.outputs_dir,
            self.figures_dir,
            self.logs_dir,
            self.reports_dir,
            self.checkpoints_dir,
            self.model_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def split_raw_dir(self, split: str) -> Path:
        """Return the image directory for a given split."""
        return self.raw_dir / split

    def processed_records_path(self, split: str) -> Path:
        """Return the canonical JSONL path for a processed split."""
        return self.processed_dir / f"{split}_records.jsonl"

    def processed_csv_path(self, split: str) -> Path:
        """Return the human-readable CSV path for a processed split."""
        return self.processed_dir / f"{split}_records.csv"

    def artifact_prefix(self, tag: str | None = None) -> str:
        """Return a filename prefix for non-baseline artifacts."""
        if not tag or tag == self.baseline_tag:
            return ""
        return f"{tag}_"

    def split_embedding_path(self, split: str, tag: str | None = None) -> Path:
        """Return the image embedding tensor path for a split."""
        return self.embeddings_dir / f"{self.artifact_prefix(tag)}{split}_image_embeddings.pt"

    def split_embedding_metadata_path(self, split: str, tag: str | None = None) -> Path:
        """Return the metadata CSV path stored next to image embeddings."""
        return self.embeddings_dir / f"{self.artifact_prefix(tag)}{split}_image_metadata.csv"

    def log_path(self) -> Path:
        """Return the shared pipeline log path."""
        return self.logs_dir / "pipeline.log"

    def schema_path(self) -> Path:
        """Return the dataset schema inspection path."""
        return self.processed_dir / "dataset_schema.json"

    def dataset_summary_report_path(self) -> Path:
        """Return the plain-text dataset summary report path."""
        return self.reports_dir / "dataset_summary.txt"

    def metrics_csv_path(self, split: str, tag: str | None = None) -> Path:
        """Return the metrics CSV path for a split."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}{split}_metrics.csv"

    def summary_txt_path(self, split: str, tag: str | None = None) -> Path:
        """Return the text summary path for a split."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}{split}_summary.txt"

    def detailed_results_path(self, split: str, tag: str | None = None) -> Path:
        """Return the per-query retrieval analysis CSV path."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}{split}_detailed_results.csv"

    def sample_queries_path(self, split: str, tag: str | None = None) -> Path:
        """Return the sample-query retrieval CSV path."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}{split}_sample_queries.csv"

    def error_analysis_path(self, split: str, tag: str | None = None) -> Path:
        """Return the failure analysis CSV path for a split."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}{split}_error_analysis.csv"

    def error_summary_path(self, split: str, tag: str | None = None) -> Path:
        """Return the plain-text failure summary path for a split."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}{split}_error_summary.txt"

    def training_history_path(self, tag: str | None = None) -> Path:
        """Return the training history CSV path."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}training_history.csv"

    def comparison_report_path(self, tag: str | None = None) -> Path:
        """Return the baseline-vs-finetuned comparison report path."""
        return self.reports_dir / f"{self.artifact_prefix(tag)}comparison_report.txt"

    def hard_negative_path(self, split: str) -> Path:
        """Return the path used to cache mined hard negatives."""
        return self.processed_dir / f"{split}_hard_negatives.json"

    def figure_path(self, filename: str, tag: str | None = None) -> Path:
        """Return a figure path with an optional experiment prefix."""
        return self.figures_dir / f"{self.artifact_prefix(tag)}{filename}"

    def best_text_checkpoint_dir(self, tag: str | None = None) -> Path:
        """Return the directory reserved for the best fine-tuned text model."""
        checkpoint_tag = tag or self.finetuned_tag
        return self.checkpoints_dir / f"{checkpoint_tag}_best_text_model"

    def latest_text_checkpoint_dir(self, tag: str | None = None) -> Path:
        """Return the directory reserved for the latest fine-tuned text model."""
        checkpoint_tag = tag or self.finetuned_tag
        return self.checkpoints_dir / f"{checkpoint_tag}_latest_text_model"

    def best_image_checkpoint_dir(self, tag: str | None = None) -> Path:
        """Return the directory reserved for the best fine-tuned image model."""
        checkpoint_tag = tag or self.finetuned_tag
        return self.checkpoints_dir / f"{checkpoint_tag}_best_image_model"

    def latest_image_checkpoint_dir(self, tag: str | None = None) -> Path:
        """Return the directory reserved for the latest fine-tuned image model."""
        checkpoint_tag = tag or self.finetuned_tag
        return self.checkpoints_dir / f"{checkpoint_tag}_latest_image_model"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return the preferred torch device string."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_logger(name: str, log_file: Path) -> logging.Logger:
    """Create or reuse a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if getattr(logger, "_is_configured", False):
        return logger

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger._is_configured = True
    return logger
