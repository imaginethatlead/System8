from __future__ import annotations

import argparse
import json
import pathlib
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


def _default_noise() -> Dict[str, List[str]]:
    return {
        "stopwords": [
            "the",
            "and",
            "or",
            "a",
            "to",
            "of",
            "for",
            "in",
            "on",
            "with",
            "is",
            "it",
            "this",
            "that",
            "my",
            "your",
            "you",
            "me",
            "we",
            "our",
        ],
        "platform": [
            "fyp",
            "tiktok",
            "viral",
            "trend",
            "like",
            "share",
            "subscribe",
            "duet",
            "stitch",
            "follow",
        ],
    }


def _default_lexicons() -> Dict[str, Dict[str, List[str]]]:
    return {
        "content_domain": {
            "beauty": ["makeup", "skincare", "grwm", "lipstick"],
            "sports": ["game", "match", "football", "basketball", "soccer"],
            "food": ["recipe", "cook", "kitchen", "restaurant"],
            "music": ["song", "album", "tour", "concert"],
        },
        "vibe": {
            "positive": ["love", "great", "fun", "happy"],
            "neutral": ["okay", "normal", "alright"],
            "negative": ["bad", "sad", "angry"],
        },
        "commercial": {
            "ad": ["sponsored", "ad", "partner", "promo"],
            "shop": ["buy", "sale", "discount", "link"],
        },
        "role": {
            "creator": ["creator", "influencer", "vlogger"],
            "professional": ["doctor", "lawyer", "chef", "teacher"],
        },
        "format": {
            "tutorial": ["tutorial", "how", "guide", "learn"],
            "review": ["review", "opinion", "rating"],
            "challenge": ["challenge", "trend", "hashtagchallenge"],
        },
        "time": {
            "seasonal": ["summer", "winter", "spring", "fall", "holiday"],
            "event": ["2024", "2025", "newyear", "christmas"],
        },
        "sentiment": {
            "positive": ["love", "great", "amazing", "good", "happy"],
            "negative": ["hate", "bad", "terrible", "sad"],
        },
    }


@dataclass
class Sys8Config:
    """Central configuration for System 8."""

    input_path: str
    output_dir: str = "outputs"
    english_only: bool = True
    language_threshold: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clustering_k: int = 25
    random_seed: int = 7
    min_anchor_coverage: float = 0.2
    max_anchor_garbage: float = 0.25
    insufficient_topic_threshold: int = 5
    hashtag_weight: float = 1.5
    lexicons: Dict[str, Dict[str, List[str]]] = field(default_factory=_default_lexicons)
    noise: Dict[str, List[str]] = field(default_factory=_default_noise)
    run_id: Optional[str] = None

    @classmethod
    def from_args(cls, args: Optional[List[str]] = None) -> "Sys8Config":
        parser = argparse.ArgumentParser(description="Run System 8 pipeline.")
        parser.add_argument("input_path", help="Input parquet path or HF dataset name")
        parser.add_argument(
            "--output-dir",
            default="outputs",
            help="Directory for run artifacts (default: outputs)",
        )
        parser.add_argument(
            "--clustering-k",
            type=int,
            default=25,
            help="Number of clusters to target for KMeans",
        )
        parser.add_argument(
            "--embedding-model",
            default="sentence-transformers/all-MiniLM-L6-v2",
            help="Sentence embedding model name",
        )
        parser.add_argument(
            "--english-only",
            action="store_true",
            default=True,
            help="Filter to English content using heuristic language detection (default on)",
        )
        parser.add_argument(
            "--allow-non-english",
            dest="english_only",
            action="store_false",
            help="Disable English-only filtering",
        )
        parser.add_argument(
            "--language-threshold",
            type=float,
            default=0.5,
            help="Minimum confidence for keeping English rows",
        )
        parser.add_argument(
            "--random-seed", type=int, default=7, help="Random seed for reproducibility"
        )
        parser.add_argument(
            "--min-anchor-coverage",
            type=float,
            default=0.2,
            help="Minimum NER anchor coverage required to publish anchors",
        )
        parser.add_argument(
            "--max-anchor-garbage",
            type=float,
            default=0.25,
            help="Maximum allowed garbage rate for anchors",
        )
        parser.add_argument(
            "--hashtag-weight",
            type=float,
            default=1.5,
            help="Explicit weighting multiplier for hashtag tokens in lexicon scoring",
        )
        parser.add_argument(
            "--insufficient-topic-threshold",
            type=int,
            default=5,
            help="Minimum token length for usable topic_text evidence",
        )
        parsed = parser.parse_args(args=args)
        return cls(
            input_path=parsed.input_path,
            output_dir=parsed.output_dir,
            clustering_k=parsed.clustering_k,
            embedding_model=parsed.embedding_model,
            english_only=parsed.english_only,
            language_threshold=parsed.language_threshold,
            random_seed=parsed.random_seed,
            min_anchor_coverage=parsed.min_anchor_coverage,
            max_anchor_garbage=parsed.max_anchor_garbage,
            hashtag_weight=parsed.hashtag_weight,
            insufficient_topic_threshold=parsed.insufficient_topic_threshold,
        )

    def ensure_run_id(self) -> str:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        return self.run_id

    def output_path(self, *parts: str) -> pathlib.Path:
        path = pathlib.Path(self.output_dir).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def save_config_snapshot(config: Sys8Config) -> None:
    path = config.output_path("sys8_config_snapshot.json")
    path.write_text(config.to_json())
