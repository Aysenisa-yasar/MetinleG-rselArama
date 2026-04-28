"""Similarity search utilities for text-to-image retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.config import ProjectConfig
from src.embedder import EmbeddingStore, MultilingualClipEmbedder, load_embedding_store


class RandomHyperplaneLSH:
    """Lightweight ANN-style index based on random hyperplane hashing."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        num_tables: int,
        num_planes: int,
        min_candidates: int,
        seed: int = 42,
    ) -> None:
        self.embeddings = embeddings.cpu()
        self.embeddings_np = self.embeddings.numpy().astype(np.float32)
        self.num_tables = num_tables
        self.num_planes = num_planes
        self.min_candidates = min_candidates
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.standard_normal(
            size=(num_tables, num_planes, self.embeddings_np.shape[1])
        ).astype(np.float32)
        self.bit_weights = (1 << np.arange(num_planes, dtype=np.uint64)).astype(np.uint64)
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(num_tables)]
        self._fit()

    def _hash_vectors(self, vectors: np.ndarray, table_index: int) -> np.ndarray:
        projections = vectors @ self.hyperplanes[table_index].T
        bits = (projections > 0).astype(np.uint64)
        return (bits * self.bit_weights).sum(axis=1)

    def _fit(self) -> None:
        for table_index in range(self.num_tables):
            hashes = self._hash_vectors(self.embeddings_np, table_index)
            table = self.tables[table_index]
            for row_index, hash_value in enumerate(hashes.tolist()):
                table[int(hash_value)].append(row_index)

    def search(self, query_embedding: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return approximate top-k scores and indices."""
        query_np = query_embedding.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        candidate_indices: set[int] = set()

        for table_index in range(self.num_tables):
            hash_value = int(self._hash_vectors(query_np, table_index)[0])
            candidate_indices.update(self.tables[table_index].get(hash_value, []))

        if len(candidate_indices) < max(top_k, self.min_candidates):
            candidate_tensor = torch.arange(len(self.embeddings))
        else:
            candidate_tensor = torch.tensor(sorted(candidate_indices), dtype=torch.long)

        candidate_embeddings = self.embeddings.index_select(0, candidate_tensor)
        scores = torch.matmul(candidate_embeddings, query_embedding.cpu())
        top_k = min(top_k, scores.numel())
        top_scores, local_indices = torch.topk(scores, k=top_k)
        top_indices = candidate_tensor.index_select(0, local_indices)
        return top_scores, top_indices


@dataclass(slots=True)
class RetrievalIndex:
    """Wrapper around normalized image embeddings and their metadata."""

    embeddings: torch.Tensor
    metadata: pd.DataFrame
    backend: str = "exact"
    ann_index: RandomHyperplaneLSH | None = None

    def search_from_text_embedding(self, text_embedding: torch.Tensor, top_k: int = 5) -> pd.DataFrame:
        """Return the top-k most similar images for a single normalized text embedding."""
        if text_embedding.ndim == 2:
            text_embedding = text_embedding.squeeze(0)

        top_k = min(top_k, len(self.metadata))
        if self.backend == "ann_lsh" and self.ann_index is not None:
            top_scores, top_indices = self.ann_index.search(text_embedding, top_k=top_k)
        else:
            scores = torch.matmul(self.embeddings, text_embedding.cpu())
            top_scores, top_indices = torch.topk(scores, k=top_k)

        rows: list[dict] = []
        for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            metadata_row = self.metadata.iloc[index]
            rows.append(
                {
                    "rank": rank,
                    "image_id": metadata_row["image_id"],
                    "image_path": metadata_row["image_path"],
                    "captions": metadata_row["captions"],
                    "score": float(score),
                }
            )
        return pd.DataFrame(rows)


def build_retrieval_index(
    split: str,
    config: ProjectConfig,
    tag: str | None = None,
    backend: str | None = None,
) -> RetrievalIndex:
    """Load a split's embeddings and wrap them as a retrieval index."""
    store: EmbeddingStore = load_embedding_store(split, config, tag=tag)
    selected_backend = backend or config.retrieval_backend
    ann_index = None
    if selected_backend == "ann_lsh":
        ann_index = RandomHyperplaneLSH(
            embeddings=store.embeddings,
            num_tables=config.lsh_num_tables,
            num_planes=config.lsh_num_planes,
            min_candidates=config.lsh_min_candidates,
            seed=config.seed,
        )
    return RetrievalIndex(
        embeddings=store.embeddings,
        metadata=store.metadata,
        backend=selected_backend,
        ann_index=ann_index,
    )


def retrieve_images(
    query: str,
    index: RetrievalIndex,
    embedder: MultilingualClipEmbedder,
    top_k: int = 5,
) -> pd.DataFrame:
    """Encode a Turkish query and return the most relevant images."""
    query_embedding = embedder.encode_texts([query])
    return index.search_from_text_embedding(query_embedding, top_k=top_k)
