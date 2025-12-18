from __future__ import annotations

import logging
import sys

from .cluster import cluster_embeddings, cluster_qa_report
from .config import Sys8Config
from .embed import embed
from .ingest import ingest
from .integrity import run_integrity
from .lexicon_scores import compute_scores
from .ner import extract_entities, ner_report, load_ner_model
from .noise import noise_report
from .operator_cards import build_operator_pack
from .prepare_evidence import prepare_evidence
from .profiles import build_profiles


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Sys8Config.from_args(argv)
    logging.info("Starting System 8 run with run_id %s", config.ensure_run_id())

    raw = ingest(config.input_path)
    prepared = prepare_evidence(raw, config)
    lex_scores = compute_scores(prepared, config)
    nlp = load_ner_model()
    mentions, entities_by_video = extract_entities(prepared, config, nlp=nlp)
    ner_metrics = ner_report(prepared, mentions, entities_by_video, config)
    embeddings = embed(prepared, config)
    clusters, coherence = cluster_embeddings(embeddings, prepared, config)
    cluster_qa_report(clusters, coherence, config)
    profiles = build_profiles(prepared, clusters, coherence, mentions, entities_by_video, config)
    run_status, _ = run_integrity(
        prepared, embeddings, entities_by_video, clusters, profiles, ner_metrics, config
    )
    noise_report(prepared, config)
    if run_status == "VALID_FOR_OPERATOR_REVIEW":
        build_operator_pack(prepared, clusters, profiles, lex_scores, run_status, config)
    else:
        logging.warning("Integrity failed; operator cards withheld.")


if __name__ == "__main__":
    main(sys.argv[1:])
