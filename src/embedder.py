"""Embedding generation and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import ProjectConfig, get_device, get_logger
from src.data_loader import load_processed_split
from src.preprocess import captions_to_json


@dataclass(slots=True)
class EmbeddingStore:
    """In-memory representation of a split's image embedding index."""

    embeddings: torch.Tensor
    metadata: pd.DataFrame


class MultilingualClipEmbedder:
    """Wraps CLIP image encoding and multilingual text encoding."""

    def __init__(
        self,
        config: ProjectConfig,
        text_model_name_or_path: str | Path | None = None,
        image_model_name_or_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.device = get_device()
        self.logger = get_logger("embedder", config.log_path())
        text_model_source = (
            str(text_model_name_or_path)
            if text_model_name_or_path is not None
            else config.text_model_name
        )
        image_model_source = (
            str(image_model_name_or_path)
            if image_model_name_or_path is not None
            else config.image_model_name
        )

        self.logger.info("Loading image encoder: %s", image_model_source)
        self.image_model = SentenceTransformer(
            image_model_source,
            device=self.device,
            cache_folder=str(config.model_cache_dir),
        )

        self.logger.info("Loading text encoder: %s", text_model_source)
        self.text_model = SentenceTransformer(
            text_model_source,
            device=self.device,
            cache_folder=str(config.model_cache_dir),
        )

    @staticmethod
    def _load_image(path: str | Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def encode_images(self, image_paths: list[str], batch_size: int | None = None) -> torch.Tensor:
        """Encode a list of image paths into normalized CLIP embeddings."""
        batch_size = batch_size or self.config.image_batch_size
        embeddings: list[torch.Tensor] = []

        for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[start_idx : start_idx + batch_size]
            images = [self._load_image(path) for path in batch_paths]
            batch_embeddings = self.image_model.encode(
                images,
                batch_size=len(images),
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> torch.Tensor:
        """Encode Turkish queries into normalized text embeddings."""
        batch_size = batch_size or self.config.text_batch_size
        return self.text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()


def save_embedding_store(
    split: str,
    embeddings: torch.Tensor,
    metadata_df: pd.DataFrame,
    config: ProjectConfig,
    tag: str | None = None,
) -> None:
    """Persist embeddings as .pt and metadata as CSV."""
    embedding_path = config.split_embedding_path(split, tag=tag)
    metadata_path = config.split_embedding_metadata_path(split, tag=tag)
    metadata_to_save = metadata_df.copy()
    metadata_to_save["captions"] = metadata_to_save["captions"].apply(captions_to_json)

    torch.save(
        {
            "split": split,
            "embeddings": embeddings.cpu(),
            "image_ids": metadata_df["image_id"].tolist(),
            "image_paths": metadata_df["image_path"].tolist(),
            "captions": metadata_df["captions"].tolist(),
            "dataset_indices": metadata_df["dataset_index"].tolist(),
        },
        embedding_path,
    )
    metadata_to_save.to_csv(metadata_path, index=False)


def load_embedding_store(split: str, config: ProjectConfig, tag: str | None = None) -> EmbeddingStore:
    """Load stored embeddings and metadata for a split."""
    embedding_path = config.split_embedding_path(split, tag=tag)
    metadata_path = config.split_embedding_metadata_path(split, tag=tag)
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}. Run the embed stage first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Embedding metadata not found: {metadata_path}. Run the embed stage first."
        )

    stored = torch.load(embedding_path, map_location="cpu")
    metadata = pd.read_csv(metadata_path)
    metadata["captions"] = metadata["captions"].apply(json.loads)
    return EmbeddingStore(embeddings=stored["embeddings"].cpu(), metadata=metadata)


def embed_split(
    split: str,
    embedder: MultilingualClipEmbedder,
    config: ProjectConfig,
    tag: str | None = None,
) -> EmbeddingStore:
    """Generate and store image embeddings for a processed split."""
    logger = get_logger("embed_split", config.log_path())
    metadata_df = load_processed_split(split, config)
    embeddings = embedder.encode_images(metadata_df["image_path"].tolist())
    save_embedding_store(split, embeddings, metadata_df, config, tag=tag)
    logger.info(
        "Saved %s embeddings for split %s to %s",
        len(metadata_df),
        split,
        config.split_embedding_path(split, tag=tag),
    )
    return EmbeddingStore(embeddings=embeddings, metadata=metadata_df)


def embed_all_splits(
    config: ProjectConfig,
    tag: str | None = None,
    text_model_name_or_path: str | Path | None = None,
    image_model_name_or_path: str | Path | None = None,
) -> dict[str, EmbeddingStore]:
    """Build image embeddings for train, validation, and test."""
    logger = get_logger("embed_all", config.log_path())
    config.ensure_directories()
    embedder = MultilingualClipEmbedder(
        config,
        text_model_name_or_path=text_model_name_or_path,
        image_model_name_or_path=image_model_name_or_path,
    )
    results: dict[str, EmbeddingStore] = {}
    for split in ("train", "validation", "test"):
        logger.info("Embedding split: %s", split)
        results[split] = embed_split(split, embedder, config, tag=tag)
    return results
