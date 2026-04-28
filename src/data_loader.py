"""Dataset loading, inspection, and standardization utilities."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from src.config import ProjectConfig, get_logger
from src.preprocess import captions_to_json, extract_captions, infer_caption_columns


def load_dataset_splits(config: ProjectConfig) -> DatasetDict:
    """Load the Turkish Flickr8k dataset from Hugging Face."""
    return load_dataset(config.dataset_name, cache_dir=str(config.model_cache_dir))


def coerce_pil_image(image_value: Any) -> Image.Image:
    """Convert a datasets Image field into an RGB PIL image."""
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")
        if image_value.get("bytes"):
            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_value)!r}")


def save_image(image_value: Any, image_path: Path) -> None:
    """Persist an image to disk if it is not already saved locally."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.exists():
        return
    pil_image = coerce_pil_image(image_value)
    pil_image.save(image_path, quality=95)


def inspect_dataset(dataset_dict: DatasetDict) -> dict[str, Any]:
    """Create a lightweight dataset schema summary."""
    summary: dict[str, Any] = {}
    for split_name, split_dataset in dataset_dict.items():
        summary[split_name] = {
            "num_rows": len(split_dataset),
            "columns": list(split_dataset.column_names),
            "caption_columns": infer_caption_columns(split_dataset.column_names),
        }
    return summary


def standardize_split(
    split_dataset,
    split_name: str,
    config: ProjectConfig,
) -> list[dict[str, Any]]:
    """Convert a dataset split into the project's standardized image-level records."""
    logger = get_logger("data_loader", config.log_path())
    caption_columns = infer_caption_columns(split_dataset.column_names)
    if not caption_columns:
        raise ValueError(
            f"No caption columns were detected for split '{split_name}'. "
            f"Available columns: {split_dataset.column_names}"
        )

    split_image_dir = config.split_raw_dir(split_name)
    split_image_dir.mkdir(parents=True, exist_ok=True)
    records_by_image_id: dict[str, dict[str, Any]] = {}

    for dataset_index, example in enumerate(tqdm(split_dataset, desc=f"Standardizing {split_name}")):
        raw_image_id = example.get("imgid", example.get("image_id", dataset_index))
        image_id = str(raw_image_id)
        image_path = split_image_dir / f"{image_id}.{config.image_format}"
        captions = extract_captions(example, caption_columns)
        save_image(example["image"], image_path)

        if image_id not in records_by_image_id:
            records_by_image_id[image_id] = {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "captions": captions,
                "split": split_name,
                "dataset_index": dataset_index,
            }
        else:
            merged_captions = records_by_image_id[image_id]["captions"] + captions
            records_by_image_id[image_id]["captions"] = list(dict.fromkeys(merged_captions))

    standardized_records = list(records_by_image_id.values())
    logger.info(
        "Split %s standardized: %s unique images, %s total captions, columns=%s",
        split_name,
        len(standardized_records),
        sum(len(record["captions"]) for record in standardized_records),
        caption_columns,
    )
    return standardized_records


def write_processed_split(records: list[dict[str, Any]], split: str, config: ProjectConfig) -> None:
    """Write processed records to JSONL and CSV."""
    jsonl_path = config.processed_records_path(split)
    csv_path = config.processed_csv_path(split)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataframe = pd.DataFrame(records).copy()
    dataframe["captions"] = dataframe["captions"].apply(captions_to_json)
    dataframe.to_csv(csv_path, index=False)


def load_processed_split(split: str, config: ProjectConfig) -> pd.DataFrame:
    """Load processed split records from JSONL."""
    jsonl_path = config.processed_records_path(split)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {jsonl_path}. Run prepare_data first."
        )

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def prepare_data(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    """Download, inspect, standardize, and persist the dataset splits."""
    logger = get_logger("prepare_data", config.log_path())
    config.ensure_directories()

    logger.info("Loading dataset: %s", config.dataset_name)
    dataset_dict = load_dataset_splits(config)
    schema = inspect_dataset(dataset_dict)

    config.schema_path().write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    processed_frames: dict[str, pd.DataFrame] = {}
    summary_lines = [f"Dataset: {config.dataset_name}", ""]

    for split_name, split_dataset in dataset_dict.items():
        records = standardize_split(split_dataset, split_name, config)
        write_processed_split(records, split_name, config)
        processed_frames[split_name] = pd.DataFrame(records)

        summary_lines.extend(
            [
                f"[{split_name}]",
                f"- Raw rows: {len(split_dataset)}",
                f"- Unique images: {len(records)}",
                f"- Total captions: {sum(len(record['captions']) for record in records)}",
                f"- Columns: {', '.join(split_dataset.column_names)}",
                f"- Caption columns: {', '.join(schema[split_name]['caption_columns'])}",
                "",
            ]
        )

    config.dataset_summary_report_path().write_text(
        "\n".join(summary_lines),
        encoding=config.report_encoding,
    )
    logger.info("Processed data saved under %s", config.processed_dir)
    return processed_frames

