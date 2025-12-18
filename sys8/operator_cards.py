from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import Sys8Config


def _recommended_action(flags: List[str]) -> str:
    if "platform_bucket" in flags or "low_anchor_coverage" in flags:
        return "quarantine_low_evidence"
    if "mixed_topic" in flags or "low_congruence" in flags:
        return "review"
    return "good"


def build_operator_pack(
    prepared: pd.DataFrame,
    clusters: pd.DataFrame,
    profiles: List[Dict],
    lexicon_scores: pd.DataFrame,
    run_status: str,
    config: Sys8Config,
) -> None:
    if run_status != "VALID_FOR_OPERATOR_REVIEW":
        logging.warning("Run invalid; skipping operator pack generation.")
        return
    score_map = lexicon_scores.set_index("video_id")
    prepared_map = prepared.set_index("video_id")
    cluster_groups = clusters.groupby("cluster_id")
    index_rows = []
    cards_dir = config.output_path("operator_cards")
    cards_dir.mkdir(parents=True, exist_ok=True)
    profile_by_id = {p["cluster_id"]: p for p in profiles}
    for cluster_id, group in cluster_groups:
        profile = profile_by_id.get(cluster_id, {})
        strengths = group["strength"]
        flags = []
        if profile.get("pct_insufficient_topic_evidence", 0) > 0.5:
            flags.append("platform_bucket")
        if profile.get("keyword_congruence", 1.0) < 0.1:
            flags.append("low_congruence")
        anchor_coverage = profile.get("anchor_coverage", 0)
        anchor_garbage = profile.get("anchor_garbage_rate", 0)
        anchor_consensus = profile.get("anchor_consensus_max", 0)
        if anchor_coverage < config.min_anchor_coverage or anchor_consensus < 0.05:
            flags.append("low_anchor_coverage")
        recommended = _recommended_action(flags)
        index_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(group),
                "coherence_score": profile.get("coherence", 0),
                "strength_median": strengths.median(),
                "strength_iqr": strengths.quantile(0.75) - strengths.quantile(0.25),
                "pct_insufficient_topic_evidence": profile.get("pct_insufficient_topic_evidence", 0),
                "keyword_congruence_score": profile.get("keyword_congruence", 0),
                "anchor_coverage": anchor_coverage,
                "anchor_garbage_rate": anchor_garbage,
                "anchor_consensus_max": anchor_consensus,
                "flags": ";".join(flags),
                "recommended_action": recommended,
            }
        )
        # Build card
        card_lines = [
            f"# Cluster {cluster_id}",
            f"Run: {config.run_id}",
            f"Size: {len(group)}",
            f"Coherence: {profile.get('coherence', 0):.3f}",
            f"Strength median/IQR: {strengths.median():.3f} / {(strengths.quantile(0.75) - strengths.quantile(0.25)):.3f}",
            f"Anchor coverage: {anchor_coverage:.2f} (consensus max {anchor_consensus:.2f})",
        ]
        card_lines.append("\n## What it's about")
        card_lines.append("### Description keywords")
        for kw, score in profile.get("keywords_desc", [])[:15]:
            card_lines.append(f"- {kw} ({score:.3f})")
        card_lines.append("### Hashtag keywords")
        for kw, score in profile.get("keywords_hashtags", [])[:15]:
            card_lines.append(f"- {kw} ({score:.3f})")

        card_lines.append("\n## Entities/Anchors (typed)")
        if anchor_coverage >= config.min_anchor_coverage and anchor_consensus >= 0.05:
            for ent_type, anchors in profile.get("anchors_by_type", {}).items():
                card_lines.append(f"- **{ent_type}**")
                for anchor in anchors:
                    card_lines.append(
                        f"  - {anchor['entity']} (coverage {anchor['coverage']:.2f}, support {anchor['support']})"
                    )
        else:
            card_lines.append("Anchors withheld due to quality thresholds.")

        card_lines.append("\n## Sentiment")
        sent_scores = []
        for vid in group["video_id"]:
            if vid in score_map.index:
                sent_scores.append(score_map.loc[vid].get("sentiment_polarity", 0))
        if sent_scores:
            card_lines.append(f"- mean polarity: {sum(sent_scores)/len(sent_scores):.3f}")
        else:
            card_lines.append("- sentiment unavailable")

        card_lines.append("\n## Lexicon overlays (top contributing tokens)")
        lex_fams = [c for c in score_map.columns if c.endswith("_top_tokens")]
        for fam_col in lex_fams:
            fam_name = fam_col.replace("_top_tokens", "")
            card_lines.append(f"- **{fam_name}**")
            token_counter = {}
            for vid in group["video_id"]:
                if vid not in score_map.index:
                    continue
                fam_tokens = score_map.loc[vid].get(fam_col, {}) or {}
                for label, pairs in fam_tokens.items():
                    for tok, weight in pairs:
                        token_counter.setdefault(label, []).append((tok, weight))
            for label, pairs in token_counter.items():
                pairs.sort(key=lambda x: x[1], reverse=True)
                snippet = ", ".join([f"{tok} ({w:.2f})" for tok, w in pairs[:5]])
                card_lines.append(f"  - {label}: {snippet}")

        card_lines.append("\n## Exemplars")
        exemplars = group["video_id"].tolist()[:20]
        for vid in exemplars:
            card_lines.append(f"- Video {vid}")
            if vid in prepared_map.index:
                row = prepared_map.loc[vid]
                card_lines.append(f"  - desc: {row['desc_raw'][:180]}")
                card_lines.append(f"  - hashtags: {' '.join(row['hashtags_list_norm'])}")
        card_path = Path(cards_dir) / f"cluster_{cluster_id}.md"
        card_path.write_text("\n".join(card_lines))
    index_df = pd.DataFrame(index_rows).sort_values("recommended_action")
    index_path = config.output_path("operator_cluster_index.csv")
    index_df.to_csv(index_path, index=False)
    logging.info("Wrote operator cluster index to %s", index_path)
