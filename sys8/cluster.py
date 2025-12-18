from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .config import Sys8Config


def _fit_kmeans(vectors: np.ndarray, k: int, seed: int) -> KMeans:
    k = min(k, len(vectors)) or 1
    return KMeans(n_clusters=k, random_state=seed, n_init=10)


def cluster_embeddings(embeddings: pd.DataFrame, prepared: pd.DataFrame, config: Sys8Config):
    vectors = np.stack(embeddings["embedding"].to_numpy())
    model = _fit_kmeans(vectors, config.clustering_k, config.random_seed)
    labels = model.fit_predict(vectors)
    sims = cosine_similarity(vectors, model.cluster_centers_)
    strengths = sims.max(axis=1)
    cluster_df = pd.DataFrame(
        {
            "run_id": embeddings["run_id"],
            "video_id": embeddings["video_id"],
            "cluster_id": labels,
            "strength": strengths,
        }
    )
    # compute coherence metrics
    coherence = {}
    insufficient_map = prepared.set_index("video_id")["insufficient_topic_evidence"].to_dict()
    for cid in np.unique(labels):
        mask = labels == cid
        cluster_vectors = vectors[mask]
        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        coh = float(cosine_similarity(cluster_vectors, centroid).mean())
        vids = embeddings["video_id"].to_numpy()[mask]
        insufficient_pct = (
            sum(bool(insufficient_map.get(v, False)) for v in vids) / len(vids) if len(vids) else 0
        )
        platform_bucket = insufficient_pct > 0.5 or coh < 0.2
        coherence[cid] = {
            "coherence": coh,
            "pct_insufficient_topic": insufficient_pct,
            "platform_bucket": platform_bucket,
        }
    cluster_df["is_noise"] = cluster_df["cluster_id"].map(
        lambda cid: coherence[cid]["platform_bucket"]
    )
    path = config.output_path("sys8_clusters.parquet")
    cluster_df.to_parquet(path, index=False)
    logging.info("Wrote clusters to %s", path)
    return cluster_df, coherence


def cluster_qa_report(cluster_df: pd.DataFrame, coherence: dict, config: Sys8Config) -> None:
    lines = [
        "# Cluster QA report",
        f"Run: {config.run_id}",
        f"Clusters: {cluster_df['cluster_id'].nunique()}",
    ]
    coh_values = [m["coherence"] for m in coherence.values()]
    if coh_values:
        lines.append(f"Coherence mean: {float(np.mean(coh_values)):.3f}")
        lines.append(f"Coherence median: {float(np.median(coh_values)):.3f}")
    lines.append("\n## Flagged clusters")
    for cid, metrics in coherence.items():
        if metrics["platform_bucket"]:
            lines.append(f"- Cluster {cid} flagged as platform bucket (coherence {metrics['coherence']:.3f}, insufficient {metrics['pct_insufficient_topic']:.2f})")
    report_path = config.output_path("sys8_cluster_qa_report.md")
    report_path.write_text("\n".join(lines))
    logging.info("Wrote cluster QA report to %s", report_path)
