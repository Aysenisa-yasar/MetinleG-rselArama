"""Preprocessing helpers for captions and standardized records."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

import pandas as pd

TURKISH_STOPWORDS = {
    "bir",
    "ve",
    "ile",
    "olan",
    "olarak",
    "icin",
    "gibi",
    "bu",
    "su",
    "da",
    "de",
    "mi",
    "mı",
    "mu",
    "mü",
    "ama",
    "veya",
    "ya",
    "ile",
    "icin",
    "ki",
    "olan",
    "olanlar",
}

SLUG_REPLACEMENTS = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "I": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    }
)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving Turkish characters and case."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def infer_caption_columns(column_names: Iterable[str]) -> list[str]:
    """Infer caption columns such as caption0, caption1, ... from dataset schema."""
    caption_columns = [name for name in column_names if name.lower().startswith("caption")]

    def sort_key(column_name: str) -> tuple[int, str]:
        suffix = column_name.lower().replace("caption", "")
        return (int(suffix) if suffix.isdigit() else 999, column_name)

    return sorted(caption_columns, key=sort_key)


def extract_captions(example: dict, caption_columns: list[str]) -> list[str]:
    """Collect non-empty captions from the detected caption columns."""
    captions: list[str] = []
    for column in caption_columns:
        value = example.get(column)
        if isinstance(value, list):
            captions.extend(clean_text(item) for item in value if clean_text(item))
        else:
            cleaned = clean_text(value)
            if cleaned:
                captions.append(cleaned)
    return list(dict.fromkeys(captions))


def captions_to_json(captions: list[str]) -> str:
    """Serialize captions into a CSV-friendly JSON string."""
    return json.dumps(captions, ensure_ascii=False)


def captions_from_json(serialized: str | list[str]) -> list[str]:
    """Deserialize captions from CSV/JSON back into a Python list."""
    if isinstance(serialized, list):
        return [clean_text(item) for item in serialized if clean_text(item)]
    if not serialized:
        return []
    parsed = json.loads(serialized)
    return [clean_text(item) for item in parsed if clean_text(item)]


def flatten_caption_records(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Expand image-level metadata into caption-level evaluation rows."""
    rows: list[dict] = []
    for row in metadata_df.to_dict(orient="records"):
        captions = captions_from_json(row["captions"])
        for caption_index, caption in enumerate(captions):
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "split": row["split"],
                    "caption_index": caption_index,
                    "caption": caption,
                }
            )
    return pd.DataFrame(rows)


def slugify_text(text: str) -> str:
    """Convert text into an ASCII-friendly filename stem."""
    normalized = clean_text(text).translate(SLUG_REPLACEMENTS).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_") or "query"


def tokenize_for_analysis(text: str) -> list[str]:
    """Tokenize text lightly for error-analysis summaries."""
    normalized = clean_text(text).translate(SLUG_REPLACEMENTS).lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return [token for token in normalized.split() if len(token) > 1]


def token_overlap(left_text: str, right_text: str) -> int:
    """Return the number of shared normalized tokens between two texts."""
    left_tokens = set(tokenize_for_analysis(left_text))
    right_tokens = set(tokenize_for_analysis(right_text))
    return len(left_tokens & right_tokens)


def top_informative_tokens(
    texts: Iterable[str],
    top_n: int = 15,
    stopwords: set[str] | None = None,
) -> list[tuple[str, int]]:
    """Return the most common non-stopword tokens from a text collection."""
    stopwords = stopwords or TURKISH_STOPWORDS
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [token for token in tokenize_for_analysis(text) if token not in stopwords]
        counter.update(tokens)
    return counter.most_common(top_n)
