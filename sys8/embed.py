from __future__ import annotations

import importlib.util
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import Sys8Config


def _sentence_transformer_encoder(model_name: str):
    spec = importlib.util.find_spec("sentence_transformers")
    if spec is None:
        logging.warning("SentenceTransformer unavailable; falling back to TF-IDF.")
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _tfidf_embeddings(texts: List[str], dim: int = 64) -> Tuple[np.ndarray, str]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    n_components = max(1, min(dim, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(X)
    return reduced, "tfidf-svd"


def embed(prepared: pd.DataFrame, config: Sys8Config) -> pd.DataFrame:
    texts = prepared["topic_text"].tolist()
    model = _sentence_transformer_encoder(config.embedding_model)
    if model:
        vectors = model.encode(texts, show_progress_bar=False)
        model_used = config.embedding_model
    else:
        vectors, model_used = _tfidf_embeddings(texts)
    embedding_list = [v.tolist() for v in vectors]
    embeddings = pd.DataFrame(
        {"run_id": prepared["run_id"], "video_id": prepared["video_id"], "embedding": embedding_list}
    )
    path = config.output_path("sys8_embeddings.parquet")
    embeddings.to_parquet(path, index=False)
    logging.info("Wrote embeddings (%s) to %s", model_used, path)
    return embeddings
