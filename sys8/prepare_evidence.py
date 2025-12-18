from __future__ import annotations

import importlib.util
import logging
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Sys8Config
from .noise import load_noise_lists, normalize_token

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@([A-Za-z0-9_.]+)")
HASHTAG_INLINE_RE = re.compile(r"#(\w+)")
TOKEN_RE = re.compile(r"[\w']+")

TEXT_VIEW_VERSION = "sys8.v1"


def _language_guess(text: str) -> Tuple[str, float]:
    spec = importlib.util.find_spec("langdetect")
    if spec is None:
        return "en", 0.51
    from langdetect import detect_langs

    langs = detect_langs(text or "")
    if not langs:
        return "unknown", 0.0
    best = langs[0]
    return best.lang, best.prob


def _clean_topic_text(desc: str, tags: List[str], noise: Dict[str, List[str]]) -> Tuple[str, bool, List[str]]:
    desc = desc or ""
    lowered = desc.lower()
    lowered = URL_RE.sub(" ", lowered)
    lowered = HANDLE_RE.sub(" ", lowered)
    lowered = HASHTAG_INLINE_RE.sub(" ", lowered)
    raw_tokens = [normalize_token(tok) for tok in TOKEN_RE.findall(lowered)]
    tokens = []
    removed_noise = []
    for t in raw_tokens:
        if t and t not in noise["stopwords"] and t not in noise["platform"]:
            tokens.append(t)
        elif t:
            removed_noise.append(t)
    hashtag_tokens = []
    for tag in tags:
        tag_norm = normalize_token(tag.lower().strip("# "))
        if not tag_norm:
            continue
        parts = re.split(r"[ _]", tag_norm)
        for part in parts:
            if part and part not in noise["stopwords"] and part not in noise["platform"]:
                hashtag_tokens.append(part)
            elif part:
                removed_noise.append(part)
    combined = tokens + hashtag_tokens
    insufficient = len(combined) < 5
    return " ".join(combined), insufficient, removed_noise


def _build_ner_text(desc: str, tags: List[str]) -> str:
    text = desc or ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub("<URL>", text)
    text = HANDLE_RE.sub(lambda m: f"<HANDLE>@{m.group(1)}</HANDLE>", text)
    formatted_tags = [f"[HASHTAG: {tag}]" for tag in tags if tag]
    return text + " " + " ".join(formatted_tags)


def _build_lex_tokens(desc: str, tags: List[str], noise: Dict[str, List[str]]) -> List[str]:
    desc = desc or ""
    desc_lower = URL_RE.sub(" ", desc.lower())
    desc_lower = HANDLE_RE.sub(" ", desc_lower)
    tokens = [normalize_token(t) for t in TOKEN_RE.findall(desc_lower)]
    tokens = [t for t in tokens if t and t not in noise["stopwords"] and t not in noise["platform"]]
    tag_tokens: List[str] = []
    for tag in tags:
        tag_norm = normalize_token(tag.lower().strip("# "))
        tag_tokens.extend([p for p in re.split(r"[ _]", tag_norm) if p])
    tag_tokens = [t for t in tag_tokens if t not in noise["stopwords"] and t not in noise["platform"]]
    return tokens + tag_tokens


def _desc_only_clean(desc: str) -> str:
    desc = desc or ""
    desc = URL_RE.sub(" ", desc)
    desc = HANDLE_RE.sub(" ", desc)
    desc = HASHTAG_INLINE_RE.sub(" ", desc)
    desc = unicodedata.normalize("NFC", desc)
    return " ".join(desc.split())


def prepare_evidence(df: pd.DataFrame, config: Sys8Config) -> pd.DataFrame:
    noise_lists = load_noise_lists(config)
    rows = []
    for _, row in df.iterrows():
        desc = row.get("desc_raw") or ""
        tags_raw = row.get("hashtags_list_raw") or []
        tags_norm = [normalize_token(t.strip("# ")) for t in tags_raw if t]
        lang, lang_conf = _language_guess(desc)
        language_keep = True
        if config.english_only:
            language_keep = lang == "en" and lang_conf >= config.language_threshold
        topic_text, insufficient_topic_evidence, removed_noise = _clean_topic_text(
            desc, tags_norm, noise_lists
        )
        ner_text = _build_ner_text(desc, tags_raw)
        lex_tokens = _build_lex_tokens(desc, tags_norm, noise_lists)
        created_at = row.get("created_at")
        try:
            created_at_val = float(created_at)
            created_date = datetime.utcfromtimestamp(created_at_val).strftime("%Y-%m-%d")
        except Exception:
            created_at_val = np.nan
            created_date = ""
        rows.append(
            {
                "run_id": config.ensure_run_id(),
                "video_id": str(row.get("video_id")),
                "author_id": str(row.get("author_id")),
                "created_at": created_at_val,
                "created_date": created_date,
                "desc_raw": desc,
                "challenges_raw": row.get("challenges_raw"),
                "hashtags_list_raw": tags_raw,
                "hashtags_list_norm": tags_norm,
                "language_pred": lang,
                "language_conf": lang_conf,
                "language_keep": language_keep,
                "topic_text": topic_text,
                "ner_text": ner_text.strip(),
                "desc_only_clean": _desc_only_clean(desc),
                "lex_tokens": lex_tokens,
                "noise_hits": removed_noise,
                "text_view_version": TEXT_VIEW_VERSION,
                "insufficient_topic_evidence": insufficient_topic_evidence,
            }
        )
    evidence = pd.DataFrame(rows)
    evidence = evidence[evidence["language_keep"]].reset_index(drop=True)
    output_path = config.output_path("sys8_prepared_evidence.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_parquet(output_path, index=False)
    logging.info("Prepared evidence rows: %d", len(evidence))
    return evidence
