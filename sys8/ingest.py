from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional, Tuple

import importlib.util

import pandas as pd

CANONICAL_COLUMNS = {
    "video_id": ["video_id", "id", "aweme_id"],
    "author_id": ["author_id", "user_id", "uid"],
    "created_at": ["created_at", "create_time", "createTime"],
    "desc_raw": ["desc", "description", "desc_raw"],
    "challenges_raw": ["challenges", "hashtags", "challenges_raw"],
}


def _choose_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def parse_challenges(value) -> Tuple[List[str], object]:
    """Normalize challenges/hashtags into a list of strings plus original."""
    original = value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [], original
    try:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                value = json.loads(text)
            else:
                # simple comma separated fall back
                return [p.strip("# ").strip() for p in text.split(",") if p.strip()], original
        if isinstance(value, dict):
            value = [value]
        tags: List[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    tags.append(item)
                elif isinstance(item, dict):
                    # common keys: title/name/challenge_name
                    for k in ["title", "name", "challenge_name"]:
                        if k in item and item[k]:
                            tags.append(str(item[k]))
                            break
        return [t for t in tags if t], original
    except Exception:
        logging.exception("Failed to parse challenges value: %s", value)
        return [], original


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for canonical, options in CANONICAL_COLUMNS.items():
        col = _choose_column(df, options)
        if col:
            renamed[col] = canonical
    normalized = df.rename(columns=renamed)
    missing = [c for c in CANONICAL_COLUMNS.keys() if c not in normalized.columns]
    for m in missing:
        normalized[m] = None
    normalized["video_id"] = normalized["video_id"].astype(str)
    if normalized["author_id"].isnull().all():
        normalized["author_id"] = "unknown"
    return normalized[list(CANONICAL_COLUMNS.keys())]


def load_dataset(input_path: str) -> pd.DataFrame:
    """Load dataset from parquet path or huggingface name if available."""
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        spec = importlib.util.find_spec("datasets")
        if spec is None:
            raise RuntimeError(
                f"Failed to load dataset {input_path}. Install datasets library or provide parquet."
            )
        from datasets import load_dataset

        ds = load_dataset(input_path, split="train")
        df = ds.to_pandas()
    logging.info("Loaded dataset with %d rows", len(df))
    return normalize_columns(df)


def ingest(input_path: str) -> pd.DataFrame:
    df = load_dataset(input_path)
    hashtags, original = [], []
    for val in df["challenges_raw"]:
        parsed, raw = parse_challenges(val)
        hashtags.append(parsed)
        original.append(raw)
    df["hashtags_list_raw"] = hashtags
    df["challenges_raw"] = original
    return df
