# SYS8 Operator Guide

Welcome to System 8. This guide describes how to inspect the outputs of a run and make review decisions safely.

## Quickstart navigation
1. **Start with `sys8_integrity_report.md`.** Confirm `Run status: VALID_FOR_OPERATOR_REVIEW` and that all gates are marked as passed. If the run is invalid, do **not** review cards; instead, read the troubleshooting tips in the report.
2. **Open `operator_cluster_index.csv`.** Sort by the `recommended_action` column to triage quickly. Clusters flagged with `platform_bucket`, `low_anchor_coverage`, or `low_congruence` should be reviewed or quarantined first.
3. **Review markdown cards in `operator_cards/cluster_<id>.md`.** Each card includes the cluster size, coherence, keywords, anchors (with quality guards), sentiment summary, lexicon evidence hints, and exemplar snippets.

## Trust indicators
- **Coherence score and strength spread** show how tight the cluster is. Low scores combined with high `pct_insufficient_topic_evidence` indicate potential platform/noise buckets.
- **Anchor coverage & garbage rate** determine whether anchors are shown. If anchors are withheld, the card will say so explicitly.
- **Keyword congruence** captures overlap between description- and hashtag-derived keywords; low congruence is a signal for mixed topics.
- **Recommended action** summarizes flags into next steps: `good`, `review`, or `quarantine_low_evidence`.

## How to quarantine platform buckets
- Buckets dominated by low evidence or platform tags (high `pct_insufficient_topic_evidence` and flag `platform_bucket`) should be tagged as `quarantine_low_evidence` in your downstream systems.
- When quarantine is selected, anchors remain withheld and exemplars should be checked only to confirm noise characteristics.

## Evidence anatomy
- **Topic text**: cleaned description + normalized hashtags (single pass) used for embeddings and clustering.
- **NER text**: case-preserving description with handles wrapped (`<HANDLE>@user</HANDLE>`), URLs replaced (`<URL>`), and structured hashtag tokens (`[HASHTAG: raw]`) to keep provenance.
- **Lexicon tokens**: cleaned, de-noised token list with explicit hashtag weighting (configurable).

## If something looks off
- Rerun the pipeline and check `sys8_integrity_report.md` troubleshooting tips.
- Inspect `sys8_ner_qa_report.md` for coverage and garbage rates; anchors are withheld automatically when quality is low.
- Use `sys8_token_noise_report.md` to adjust stopword/platform-noise lists when keywords drift toward noise.
