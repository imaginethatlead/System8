"""
System 8 clean-room analysis package.

The modules inside this package implement the end-to-end pipeline described in
the project brief. The public entrypoint is ``sys8.run.main``.
"""

__all__ = [
    "config",
    "ingest",
    "prepare_evidence",
    "noise",
    "lexicon_scores",
    "ner",
    "embed",
    "cluster",
    "profiles",
    "integrity",
    "operator_cards",
]
