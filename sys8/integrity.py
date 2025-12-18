from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Dict, List, Tuple

import pandas as pd

from .config import Sys8Config, save_config_snapshot


def _universe_hash(series: pd.Series) -> str:
    joined = "|".join(sorted(series.astype(str).tolist()))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _gate_sample_ids(profiles: List[Dict], prepared_ids: set, cluster_map: Dict[str, int]) -> float:
    if not profiles:
        return 0.0
    sampled = random.sample(profiles, min(5, len(profiles)))
    matches, total = 0, 0
    for profile in sampled:
        for vid in profile["sample_video_ids"]:
            total += 1
            if vid in prepared_ids and cluster_map.get(vid) == profile["cluster_id"]:
                matches += 1
    return matches / total if total else 0.0


def run_integrity(
    prepared: pd.DataFrame,
    embeddings: pd.DataFrame,
    entities_by_video: pd.DataFrame,
    clusters: pd.DataFrame,
    profiles: List[Dict],
    ner_metrics: Dict[str, float],
    config: Sys8Config,
) -> Tuple[str, Dict[str, float]]:
    random.seed(config.random_seed)
    save_config_snapshot(config)
    manifest = {
        "run_id": config.run_id,
        "output_dir": config.output_dir,
        "row_counts": {
            "prepared": len(prepared),
            "embeddings": len(embeddings),
            "entities_by_video": len(entities_by_video),
            "clusters": len(clusters),
        },
        "config": json.loads(config.to_json()),
    }
    prepared_hash = _universe_hash(prepared["video_id"])
    hashes_match = all(
        [
            prepared_hash == _universe_hash(embeddings["video_id"]),
            prepared_hash == _universe_hash(entities_by_video["video_id"]),
            prepared_hash == _universe_hash(clusters["video_id"]),
        ]
    )
    sample_video_rate = _gate_sample_ids(
        profiles, set(prepared["video_id"]), clusters.set_index("video_id")["cluster_id"].to_dict()
    )
    ner_coverage = ner_metrics.get("coverage", 0.0)
    run_status = "VALID_FOR_OPERATOR_REVIEW"
    gate_failures = []
    if not hashes_match:
        gate_failures.append("Gate1: universe hash mismatch")
    if sample_video_rate < 0.9:
        gate_failures.append("Gate2: sample video ids invalid")
    if gate_failures:
        run_status = "INVALID_FOR_OPERATOR_REVIEW"
    manifest["run_status"] = run_status
    manifest_path = config.output_path("sys8_run_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    lines = [
        "# Integrity report",
        f"Run: {config.run_id}",
        f"Run status: {run_status}",
        f"Universe hash: {prepared_hash}",
        f"Hashes consistent: {hashes_match}",
        f"Sample video id pass rate: {sample_video_rate:.2f}",
        f"NER coverage: {ner_coverage:.2f}",
        "",
        "## Troubleshooting",
        "- NER coverage low -> verify NER ran on same universe + no missing video_id join keys",
        "- sample ids missing -> verify profiles are built from clusters output, not cached/stale",
        "- keyword congruence low -> inspect noise lists and TF-IDF keywording",
        "- many platform buckets -> inspect topic_text cleaning and insufficient_topic_evidence thresholds",
        "",
        "## Gate failures" if gate_failures else "## All gates passed",
    ]
    for failure in gate_failures:
        lines.append(f"- {failure}")
    report_path = config.output_path("sys8_integrity_report.md")
    report_path.write_text("\n".join(lines))
    logging.info("Wrote integrity report to %s", report_path)
    metrics = {
        "sample_video_rate": sample_video_rate,
        "hashes_match": hashes_match,
        "ner_coverage": ner_coverage,
    }
    return run_status, metrics
