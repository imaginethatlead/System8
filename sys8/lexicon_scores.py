from __future__ import annotations

import collections
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Sys8Config
from .noise import is_noise_token, load_noise_lists

ScoreResult = Tuple[Dict[str, float], Dict[str, List[Tuple[str, float]]]]


def _score_family(tokens: List[str], lexicon: Dict[str, List[str]], weight: float) -> ScoreResult:
    scores: Dict[str, float] = {}
    evidence: Dict[str, List[Tuple[str, float]]] = {}
    token_counts = collections.Counter(tokens)
    for label, keywords in lexicon.items():
        label_score = 0.0
        contributions: List[Tuple[str, float]] = []
        for kw in keywords:
            kw_norm = kw.lower()
            tok_score = token_counts.get(kw_norm, 0) * weight
            if tok_score:
                contributions.append((kw_norm, tok_score))
            label_score += tok_score
        scores[label] = label_score
        contributions.sort(key=lambda x: x[1], reverse=True)
        evidence[label] = contributions[:5]
    return scores, evidence


def compute_scores(evidence_df: pd.DataFrame, config: Sys8Config) -> pd.DataFrame:
    noise = load_noise_lists(config)
    rows = []
    for _, row in evidence_df.iterrows():
        base_tokens = [t for t in row["lex_tokens"] if not is_noise_token(t, noise)]
        hashtag_tokens = [t for t in row["hashtags_list_norm"] if t]
        tokens = list(base_tokens)
        hashtag_repeat = max(1, int(round(config.hashtag_weight)))
        tokens.extend(hashtag_tokens * hashtag_repeat)
        entry = {
            "run_id": row["run_id"],
            "video_id": row["video_id"],
        }
        evidence_payload = {}
        sentiment_score = 0.0
        for family, lexicon in config.lexicons.items():
            scores, top_tokens = _score_family(tokens, lexicon, weight=1.0)
            entry[f"{family}_scores_raw"] = scores
            entry[f"{family}_top_tokens"] = {
                label: contrib for label, contrib in top_tokens.items() if contrib
            }
            evidence_payload[family] = top_tokens
            if family == "sentiment":
                sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)
        entry["sentiment_polarity"] = sentiment_score
        entry["lexicon_evidence"] = evidence_payload
        rows.append(entry)
    lex_df = pd.DataFrame(rows)
    path = config.output_path("sys8_lexicon_scores.parquet")
    lex_df.to_parquet(path, index=False)
    logging.info("Wrote lexicon scores with %d rows", len(lex_df))
    return lex_df
