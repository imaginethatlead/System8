# System 8 Changelog

## Files created
- `sys8/` package implementing ingest, evidence preparation, noise handling, lexicon scoring, NER, embeddings, clustering, profiles, integrity checks, and operator card generation.
- `SYS8_OPERATOR_GUIDE.md` describing operator review flow.
- `sys8` run artifacts (written on execution) including prepared evidence, embeddings, lexicon scores, NER outputs, cluster assignments, profiles, QA reports, integrity manifest, and operator pack assets.

## Stage-by-stage expectations
Row counts are captured automatically per run in `sys8_run_manifest.json`:
- **Ingest**: count of raw rows loaded from parquet/HF dataset.
- **Prepare evidence**: language-filtered rows, with insufficient-topic-evidence tracking.
- **Lexicon scores**: one row per kept video with token-evidence fields.
- **NER**: mention-level and per-video aggregates.
- **Embeddings**: vector per video_id using topic_text.
- **Clustering**: cluster assignments with membership strength.
- **Profiles**: TF-IDF keywords, anchors, sample ids per cluster.

## Integrity gate results
The integrity report (`sys8_integrity_report.md`) records:
- Gate 1: universe hash consistency across prepared evidence, embeddings, entities_by_video, and clusters.
- Gate 2: sample_video_id validity rate (target ≥0.90).
- Gate 3: NER join coverage (target ≥0.90).
- Gate 4: anchor quality thresholds before publishing.
`run_status` is set to `INVALID_FOR_OPERATOR_REVIEW` if any gate fails, and operator cards are withheld.

## NER coverage and garbage metrics
`sys8_ner_qa_report.md` captures coverage, garbage rate, and handle mis-typing rate. Anchors are withheld on low coverage or high garbage automatically; handle mistypes are prevented by construction via `<HANDLE>` tagging and strict filtering.

## What improved vs. prior baseline
- Hard integrity gates block unusable operator artifacts.
- Evidence-first views (topic_text, ner_text, lex_tokens) prevent reused blobs.
- Explicit hashtag weighting and token evidence make lexicon scores explainable.
- Anchors are typed, canonicalized, and withheld when coverage/garbage thresholds are not met.
- Operator pack surfaces trust indicators (coherence, evidence sufficiency, anchor quality, keyword congruence) to simplify review.
