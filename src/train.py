"""Improved fine-tuning pipeline with hard negatives and partial vision unfreezing."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import ProjectConfig, get_device, get_logger, set_seed
from src.data_loader import load_processed_split, prepare_data
from src.embedder import embed_all_splits, load_embedding_store
from src.evaluate import compute_retrieval_details, evaluate_split
from src.preprocess import clean_text, flatten_caption_records


class ImageCaptionContrastiveDataset(Dataset):
    """Returns one positive image-caption pair and optional mined hard negatives."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        hard_negative_map: dict[str, list[str]],
        image_id_to_path: dict[str, str],
        seed: int,
        hard_negative_count: int,
        sample_random_caption: bool = True,
    ) -> None:
        self.records = metadata_df.to_dict(orient="records")
        self.hard_negative_map = hard_negative_map
        self.image_id_to_path = image_id_to_path
        self.rng = random.Random(seed)
        self.hard_negative_count = hard_negative_count
        self.sample_random_caption = sample_random_caption

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        captions = [clean_text(caption) for caption in row["captions"] if clean_text(caption)]
        if not captions:
            raise ValueError(f"No valid captions found for image_id={row['image_id']}")

        caption = self.rng.choice(captions) if self.sample_random_caption else captions[0]
        hard_negative_candidates = self.hard_negative_map.get(str(row["image_id"]), [])
        if self.hard_negative_count > 0 and hard_negative_candidates:
            sampled_negative_ids = self.rng.sample(
                hard_negative_candidates,
                k=min(self.hard_negative_count, len(hard_negative_candidates)),
            )
        else:
            sampled_negative_ids = []

        hard_negative_paths = [
            self.image_id_to_path[negative_id]
            for negative_id in sampled_negative_ids
            if negative_id in self.image_id_to_path
        ]
        return {
            "caption": caption,
            "image_id": str(row["image_id"]),
            "image_path": row["image_path"],
            "hard_negative_paths": hard_negative_paths,
        }


class MultimodalBatchCollator:
    """Tokenizes text and preprocesses positive/negative images for CLIP."""

    def __init__(self, text_model: SentenceTransformer, image_model: SentenceTransformer) -> None:
        self.text_model = text_model
        self.image_processor = image_model[0].processor

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __call__(self, batch: list[dict]) -> dict:
        texts = [item["caption"] for item in batch]
        text_features = self.text_model.preprocess(texts)

        positive_images = [self._load_image(item["image_path"]) for item in batch]
        positive_pixel_values = self.image_processor(
            images=positive_images,
            return_tensors="pt",
        )["pixel_values"]

        negative_owner_indices: list[int] = []
        negative_images: list[Image.Image] = []
        for sample_index, item in enumerate(batch):
            for negative_path in item["hard_negative_paths"]:
                negative_images.append(self._load_image(negative_path))
                negative_owner_indices.append(sample_index)

        negative_pixel_values = None
        if negative_images:
            negative_pixel_values = self.image_processor(
                images=negative_images,
                return_tensors="pt",
            )["pixel_values"]

        return {
            "features": text_features,
            "positive_pixel_values": positive_pixel_values,
            "negative_pixel_values": negative_pixel_values,
            "negative_owner_indices": torch.tensor(negative_owner_indices, dtype=torch.long),
        }


@dataclass(slots=True)
class TrainingOutputs:
    """Container for training and final evaluation metrics."""

    best_score: float
    best_val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    history: list[dict[str, float]]


def _ensure_training_prerequisites(config: ProjectConfig) -> None:
    """Prepare required processed data and baseline image embeddings."""
    if not config.processed_records_path("train").exists():
        prepare_data(config)
    if not config.split_embedding_path("train").exists():
        embed_all_splits(config)


def _freeze_all_parameters(model: SentenceTransformer) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _configure_text_unfreezing(text_model: SentenceTransformer, layers_to_unfreeze: int) -> None:
    """Unfreeze the projection head and last text transformer layers."""
    _freeze_all_parameters(text_model)

    if len(text_model) > 1:
        for parameter in text_model[1].parameters():
            parameter.requires_grad = True
    if len(text_model) > 2:
        for parameter in text_model[2].parameters():
            parameter.requires_grad = True

    transformer = text_model[0].auto_model
    available_layers = len(transformer.transformer.layer)
    layers_to_unfreeze = min(max(layers_to_unfreeze, 0), available_layers)
    for layer in transformer.transformer.layer[-layers_to_unfreeze:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def _configure_vision_unfreezing(image_model: SentenceTransformer, layers_to_unfreeze: int) -> None:
    """Freeze CLIP vision tower except the projection head and last ViT blocks."""
    _freeze_all_parameters(image_model)
    clip_model = image_model[0].auto_model

    for parameter in clip_model.visual_projection.parameters():
        parameter.requires_grad = True
    for parameter in clip_model.vision_model.post_layernorm.parameters():
        parameter.requires_grad = True
    for parameter in clip_model.vision_model.pre_layrnorm.parameters():
        parameter.requires_grad = True

    available_layers = len(clip_model.vision_model.encoder.layers)
    layers_to_unfreeze = min(max(layers_to_unfreeze, 0), available_layers)
    for layer in clip_model.vision_model.encoder.layers[-layers_to_unfreeze:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def _count_trainable_parameters(model: SentenceTransformer) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _move_features_to_device(features: dict, device: str) -> dict:
    moved_features: dict = {}
    for key, value in features.items():
        moved_features[key] = value.to(device) if hasattr(value, "to") else value
    return moved_features


def _forward_text_embeddings(
    text_model: SentenceTransformer,
    features: dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    moved_features = _move_features_to_device(features, device)
    outputs = text_model(moved_features)
    return F.normalize(outputs["sentence_embedding"], p=2, dim=1)


def _forward_image_embeddings(
    image_model: SentenceTransformer,
    pixel_values: torch.Tensor,
    device: str,
) -> torch.Tensor:
    clip_model = image_model[0].auto_model
    image_features = clip_model.get_image_features(pixel_values=pixel_values.to(device))
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output
    return F.normalize(image_features, p=2, dim=1)


def compute_training_losses(
    text_model: SentenceTransformer,
    image_model: SentenceTransformer,
    batch: dict,
    device: str,
    config: ProjectConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute in-batch contrastive loss plus an explicit hard-negative ranking loss."""
    text_embeddings = _forward_text_embeddings(text_model, batch["features"], device)
    positive_image_embeddings = _forward_image_embeddings(
        image_model,
        batch["positive_pixel_values"],
        device,
    )

    logits = torch.matmul(text_embeddings, positive_image_embeddings.T) / config.temperature
    labels = torch.arange(logits.size(0), device=device)
    text_to_image_loss = F.cross_entropy(logits, labels)
    image_to_text_loss = F.cross_entropy(logits.T, labels)
    in_batch_loss = 0.5 * (text_to_image_loss + image_to_text_loss)

    hard_negative_loss = torch.tensor(0.0, device=device)
    negative_pixel_values = batch["negative_pixel_values"]
    negative_owner_indices = batch["negative_owner_indices"]
    if negative_pixel_values is not None and negative_owner_indices.numel() > 0:
        negative_image_embeddings = _forward_image_embeddings(
            image_model,
            negative_pixel_values,
            device,
        )
        negative_owner_indices = negative_owner_indices.to(device)
        positive_scores = (text_embeddings * positive_image_embeddings).sum(dim=1)
        negative_scores = (
            text_embeddings.index_select(0, negative_owner_indices) * negative_image_embeddings
        ).sum(dim=1)
        hard_negative_loss = F.relu(
            config.hard_negative_margin
            + negative_scores
            - positive_scores.index_select(0, negative_owner_indices)
        ).mean()

    total_loss = in_batch_loss + config.hard_negative_weight * hard_negative_loss
    return total_loss, {
        "total_loss": float(total_loss.detach().item()),
        "in_batch_loss": float(in_batch_loss.detach().item()),
        "hard_negative_loss": float(hard_negative_loss.detach().item()),
    }


def _encode_image_paths_for_eval(
    image_model: SentenceTransformer,
    image_paths: list[str],
    batch_size: int,
) -> torch.Tensor:
    """Encode images with the current image model for validation/test evaluation."""
    embeddings: list[torch.Tensor] = []
    for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Encoding eval images", leave=False):
        batch_paths = image_paths[start_idx : start_idx + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        batch_embeddings = image_model.encode(
            images,
            batch_size=len(images),
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


def compute_validation_metrics(
    text_model: SentenceTransformer,
    image_model: SentenceTransformer,
    split: str,
    config: ProjectConfig,
) -> dict[str, float]:
    """Compute retrieval metrics on a validation split using the current models."""
    metadata_df = load_processed_split(split, config)
    queries_df = flatten_caption_records(metadata_df)

    text_model.eval()
    image_model.eval()
    with torch.no_grad():
        query_embeddings = text_model.encode(
            queries_df["caption"].tolist(),
            batch_size=config.text_batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()
        image_embeddings = _encode_image_paths_for_eval(
            image_model=image_model,
            image_paths=metadata_df["image_path"].tolist(),
            batch_size=config.image_batch_size,
        )

    _, metrics = compute_retrieval_details(
        queries_df=queries_df,
        image_metadata=metadata_df,
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        top_k=config.max_eval_top_k,
    )
    return metrics


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "mrr":
        return float(metrics["MRR"])
    return float(metrics["AvgRecall"])


def _save_training_state(
    output_dir: Path,
    history: list[dict[str, float]],
    extra_state: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"history": history, **extra_state}
    (output_dir / "training_state.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_or_mine_hard_negatives(
    config: ProjectConfig,
    metadata_df: pd.DataFrame,
    text_model: SentenceTransformer,
) -> dict[str, list[str]]:
    """Load cached hard negatives or mine them from the baseline embedding space."""
    cache_path = config.hard_negative_path("train")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    image_store = load_embedding_store("train", config)
    image_embeddings = image_store.embeddings
    joined_captions = [
        " ".join(clean_text(caption) for caption in row["captions"] if clean_text(caption))
        for row in metadata_df.to_dict(orient="records")
    ]

    text_model.eval()
    with torch.no_grad():
        text_embeddings = text_model.encode(
            joined_captions,
            batch_size=config.text_batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()

    text_scores = torch.matmul(text_embeddings, image_embeddings.T)
    image_scores = torch.matmul(image_embeddings, image_embeddings.T)
    image_ids = [str(image_id) for image_id in metadata_df["image_id"].tolist()]
    hard_negative_map: dict[str, list[str]] = {}

    for row_index, image_id in enumerate(image_ids):
        candidate_ids: list[str] = []
        for score_tensor in (text_scores[row_index], image_scores[row_index]):
            ranking = torch.argsort(score_tensor, descending=True).tolist()
            for candidate_index in ranking:
                candidate_image_id = image_ids[candidate_index]
                if candidate_image_id == image_id or candidate_image_id in candidate_ids:
                    continue
                candidate_ids.append(candidate_image_id)
                if len(candidate_ids) >= config.hard_negative_pool_size:
                    break
            if len(candidate_ids) >= config.hard_negative_pool_size:
                break

        hard_negative_map[image_id] = candidate_ids[: config.hard_negative_pool_size]

    cache_path.write_text(
        json.dumps(hard_negative_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return hard_negative_map


def _load_metrics_from_csv(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    metrics_df = pd.read_csv(path)
    return {row["metric"]: float(row["value"]) for _, row in metrics_df.iterrows()}


def _write_comparison_report(
    baseline_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    output_path: Path,
) -> None:
    lines = ["Baseline vs Finetuned Comparison", ""]
    for metric in ("Recall@1", "Recall@5", "Recall@10", "MRR", "AvgRecall"):
        baseline_value = baseline_metrics.get(metric, 0.0)
        finetuned_value = finetuned_metrics.get(metric, 0.0)
        delta = finetuned_value - baseline_value
        lines.append(
            f"{metric}: baseline={baseline_value:.4f} | finetuned={finetuned_value:.4f} | delta={delta:+.4f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_text_encoder(config: ProjectConfig) -> TrainingOutputs:
    """Fine-tune the multilingual text encoder and partially unfreeze the CLIP vision tower."""
    logger = get_logger("train", config.log_path())
    config.ensure_directories()
    set_seed(config.seed)
    _ensure_training_prerequisites(config)

    device = get_device()
    logger.info("Using device: %s", device)
    logger.info("Loading text model: %s", config.text_model_name)
    logger.info("Loading image model: %s", config.image_model_name)
    text_model = SentenceTransformer(
        config.text_model_name,
        device=device,
        cache_folder=str(config.model_cache_dir),
    )
    image_model = SentenceTransformer(
        config.image_model_name,
        device=device,
        cache_folder=str(config.model_cache_dir),
    )
    text_model.to(device)
    image_model.to(device)

    _configure_text_unfreezing(text_model, config.text_unfreeze_layers)
    _configure_vision_unfreezing(image_model, config.vision_unfreeze_layers)
    logger.info("Trainable text parameters: %s", _count_trainable_parameters(text_model))
    logger.info("Trainable image parameters: %s", _count_trainable_parameters(image_model))

    train_metadata = load_processed_split("train", config)
    image_id_to_path = {
        str(row["image_id"]): row["image_path"] for row in train_metadata.to_dict(orient="records")
    }
    hard_negative_map = _load_or_mine_hard_negatives(config, train_metadata, text_model)

    train_dataset = ImageCaptionContrastiveDataset(
        metadata_df=train_metadata,
        hard_negative_map=hard_negative_map,
        image_id_to_path=image_id_to_path,
        seed=config.seed,
        hard_negative_count=config.hard_negative_count,
        sample_random_caption=True,
    )
    collator = MultimodalBatchCollator(text_model=text_model, image_model=image_model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    optimizer = AdamW(
        [
            {
                "params": [parameter for parameter in text_model.parameters() if parameter.requires_grad],
                "lr": config.text_learning_rate,
            },
            {
                "params": [parameter for parameter in image_model.parameters() if parameter.requires_grad],
                "lr": config.vision_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    total_steps = max(len(train_loader) * config.num_epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_score = float("-inf")
    best_val_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []

    initial_val_metrics = compute_validation_metrics(text_model, image_model, "validation", config)
    logger.info("Initial validation metrics: %s", initial_val_metrics)

    for epoch in range(1, config.num_epochs + 1):
        logger.info("Starting epoch %s / %s", epoch, config.num_epochs)
        text_model.train()
        image_model.train()
        epoch_losses: list[float] = []
        epoch_in_batch_losses: list[float] = []
        epoch_hard_negative_losses: list[float] = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = compute_training_losses(
                text_model=text_model,
                image_model=image_model,
                batch=batch,
                device=device,
                config=config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(image_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss_dict["total_loss"])
            epoch_in_batch_losses.append(loss_dict["in_batch_loss"])
            epoch_hard_negative_losses.append(loss_dict["hard_negative_loss"])
            progress_bar.set_postfix(
                total=f"{loss_dict['total_loss']:.4f}",
                hard=f"{loss_dict['hard_negative_loss']:.4f}",
            )

        val_metrics = compute_validation_metrics(text_model, image_model, "validation", config)
        selection_score = _selection_score(val_metrics, config.checkpoint_selection_metric)
        epoch_summary = {
            "epoch": float(epoch),
            "train_total_loss": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
            "train_in_batch_loss": float(sum(epoch_in_batch_losses) / max(len(epoch_in_batch_losses), 1)),
            "train_hard_negative_loss": float(
                sum(epoch_hard_negative_losses) / max(len(epoch_hard_negative_losses), 1)
            ),
            **{metric: float(value) for metric, value in val_metrics.items()},
            "selection_score": float(selection_score),
        }
        history.append(epoch_summary)

        latest_text_dir = config.latest_text_checkpoint_dir(config.finetuned_tag)
        latest_image_dir = config.latest_image_checkpoint_dir(config.finetuned_tag)
        text_model.save(str(latest_text_dir))
        image_model.save(str(latest_image_dir))
        _save_training_state(
            output_dir=config.checkpoints_dir / f"{config.finetuned_tag}_latest_state",
            history=history,
            extra_state={"best_score": best_score, "latest_val_metrics": val_metrics},
        )

        if selection_score > best_score:
            best_score = selection_score
            best_val_metrics = val_metrics
            text_model.save(str(config.best_text_checkpoint_dir(config.finetuned_tag)))
            image_model.save(str(config.best_image_checkpoint_dir(config.finetuned_tag)))
            _save_training_state(
                output_dir=config.checkpoints_dir / f"{config.finetuned_tag}_best_state",
                history=history,
                extra_state={"best_score": best_score, "best_val_metrics": best_val_metrics},
            )

        logger.info(
            "Epoch %s finished | total_loss=%.4f | val_metrics=%s",
            epoch,
            epoch_summary["train_total_loss"],
            val_metrics,
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(config.training_history_path(config.finetuned_tag), index=False)

    logger.info("Running final evaluation with best fine-tuned checkpoints")
    finetuned_test_metrics = evaluate_split(
        config=config,
        split="test",
        top_k=config.max_eval_top_k,
        text_model_name_or_path=str(config.best_text_checkpoint_dir(config.finetuned_tag)),
        image_model_name_or_path=str(config.best_image_checkpoint_dir(config.finetuned_tag)),
        tag=config.finetuned_tag,
    )

    baseline_metrics = _load_metrics_from_csv(config.metrics_csv_path("test"))
    if baseline_metrics is None or "AvgRecall" not in baseline_metrics:
        baseline_metrics = evaluate_split(config=config, split="test", top_k=config.max_eval_top_k)

    _write_comparison_report(
        baseline_metrics=baseline_metrics,
        finetuned_metrics=finetuned_test_metrics,
        output_path=config.comparison_report_path(config.finetuned_tag),
    )

    return TrainingOutputs(
        best_score=best_score,
        best_val_metrics=best_val_metrics,
        test_metrics=finetuned_test_metrics,
        history=history,
    )
