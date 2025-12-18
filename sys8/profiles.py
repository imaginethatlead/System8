from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Sys8Config


def _top_keywords(texts: List[str], limit: int = 15) -> List[Tuple[str, float]]:
    if not texts:
        return []
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    pairs = list(zip(vocab, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:limit]


def build_profiles(
    prepared: pd.DataFrame,
    clusters: pd.DataFrame,
    coherence: Dict[int, Dict],
    mentions: pd.DataFrame,
    entities_by_video: pd.DataFrame,
    config: Sys8Config,
) -> List[Dict]:
    prepared_map = prepared.set_index("video_id")
    cluster_groups = clusters.groupby("cluster_id")
    mentions_by_cluster = mentions.groupby("video_id")
    profile_list = []
    for cluster_id, group in cluster_groups:
        member_ids = group["video_id"].tolist()
        desc_texts = [prepared_map.loc[vid, "topic_text"] for vid in member_ids]
        desc_only_texts = [prepared_map.loc[vid, "desc_only_clean"] for vid in member_ids]
        hashtag_texts = [" ".join(prepared_map.loc[vid, "hashtags_list_norm"]) for vid in member_ids]
        keywords_desc = _top_keywords(desc_only_texts)
        keywords_hashtags = _top_keywords(hashtag_texts)
        # anchors
        anchor_counter = defaultdict(Counter)
        videos_with_anchors = set()
        for vid in member_ids:
            if vid in mentions_by_cluster.groups:
                videos_with_anchors.add(vid)
                for _, mention in mentions_by_cluster.get_group(vid).iterrows():
                    anchor_counter[mention["entity_type"]][mention["entity_canon"]] += 1
        anchors_by_type = {}
        garbage_counts = 0
        total_anchor_surfaces = 0
        consensus_max = 0.0
        for ent_type, counter in anchor_counter.items():
            total_support = sum(counter.values())
            anchors = []
            for canon, count in counter.most_common(10):
                coverage = count / len(member_ids)
                total_anchor_surfaces += 1
                consensus_max = max(consensus_max, coverage)
                anchors.append(
                    {"entity": canon, "coverage": coverage, "support": count, "type": ent_type}
                )
            anchors_by_type[ent_type] = anchors
        insufficient_pct = coherence[cluster_id]["pct_insufficient_topic"]
        anchor_coverage = len(videos_with_anchors) / len(member_ids) if member_ids else 0
        anchor_garbage = 0.0
        anchor_consensus = consensus_max
        sample_ids = member_ids[: min(20, len(member_ids))]
        keyword_congruence = 0.0
        if keywords_desc and keywords_hashtags:
            desc_set = {k for k, _ in keywords_desc}
            hash_set = {k for k, _ in keywords_hashtags}
            keyword_congruence = len(desc_set & hash_set) / max(len(desc_set | hash_set), 1)
        profile = {
            "run_id": config.run_id,
            "cluster_id": int(cluster_id),
            "size": len(member_ids),
            "coherence": coherence[cluster_id]["coherence"],
            "pct_insufficient_topic_evidence": insufficient_pct,
            "keywords_desc": keywords_desc,
            "keywords_hashtags": keywords_hashtags,
            "anchors_by_type": anchors_by_type,
            "sample_video_ids": sample_ids,
            "anchor_coverage": anchor_coverage,
            "anchor_garbage_rate": anchor_garbage,
            "anchor_consensus_max": anchor_consensus,
            "keyword_congruence": keyword_congruence,
            "median_topic_length": float(np.median([len(t.split()) for t in desc_texts]) if desc_texts else 0),
        }
        profile_list.append(profile)
    path = config.output_path("sys8_cluster_profiles.json")
    path.write_text(json.dumps(profile_list, indent=2))
    logging.info("Wrote cluster profiles to %s", path)
    return profile_list
