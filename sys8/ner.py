from __future__ import annotations

import importlib.util
import logging
import re
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd

from .config import Sys8Config
from .noise import is_noise_token, load_noise_lists, normalize_token

HANDLE_TAG_RE = re.compile(r"<HANDLE>@([A-Za-z0-9_.]+)</HANDLE>")
PUNCT_RE = re.compile(r"^[\W_]+$")
DIGIT_RE = re.compile(r"^\d+$")


def _canon(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _looks_like_entity(tag: str) -> bool:
    lowered = tag.lower()
    entity_markers = ["official", "inc", "corp", "brand", "tv", "fm", "records"]
    topic_suffix = ["tok", "tips", "life", "challenge", "tutorial", "grwm"]
    if any(lowered.endswith(s) for s in topic_suffix):
        return False
    return any(mark in lowered for mark in entity_markers) or lowered.istitle() or lowered.isupper()


def _reject(entity: str, entity_type: str, noise: Dict[str, List[str]]) -> Tuple[bool, str]:
    if not entity or PUNCT_RE.match(entity):
        return True, "punct"
    if DIGIT_RE.match(entity):
        return True, "digit_only"
    if "@" in entity and entity_type != "HANDLE":
        return True, "handle_mistype"
    if is_noise_token(normalize_token(entity), noise):
        return True, "noise"
    if len(entity) > 80:
        return True, "too_long"
    return False, ""


def load_ner_model() -> Optional[Any]:
    spec = importlib.util.find_spec("spacy")
    if spec is None:
        logging.warning("spaCy not available; falling back to regex heuristics.")
        return None
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner")
        return nlp


def _desc_model_entities(text: str, nlp) -> List[Tuple[str, str, float]]:
    if nlp is None:
        ents = []
        for match in re.finditer(r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text):
            ents.append((match.group(1), "PERSON", 0.5))
        return ents
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ent_type = ent.label_
        mapped = {
            "PERSON": "PERSON",
            "ORG": "ORG_BRAND",
            "GPE": "GPE_LOC",
            "LOC": "GPE_LOC",
            "WORK_OF_ART": "WORK",
            "PRODUCT": "PRODUCT",
        }.get(ent_type, None)
        if mapped:
            try:
                conf_val = float(ent.kb_id_) if ent.kb_id_ else 0.8
            except Exception:
                conf_val = 0.8
            ents.append((ent.text, mapped, conf_val))
    return ents


def extract_entities(
    prepared: pd.DataFrame, config: Sys8Config, nlp=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    noise = load_noise_lists(config)
    mentions_rows = []
    video_entities = defaultdict(lambda: defaultdict(list))
    reject_reasons = defaultdict(Counter)

    for _, row in prepared.iterrows():
        video_id = row["video_id"]
        run_id = row["run_id"]
        ner_text = row["ner_text"] or ""
        # Handles
        for handle in HANDLE_TAG_RE.findall(ner_text):
            surface = f"@{handle}"
            canon = _canon(surface)
            reject, reason = _reject(canon, "HANDLE", noise)
            if reject:
                reject_reasons[video_id][reason] += 1
                continue
            mentions_rows.append(
                {
                    "run_id": run_id,
                    "video_id": video_id,
                    "entity_surface": surface,
                    "entity_canon": canon.lower(),
                    "entity_type": "HANDLE",
                    "source": "HANDLE_REGEX",
                    "confidence": 0.99,
                    "evidence_snippet": surface,
                }
            )
            video_entities[video_id]["HANDLE"].append(canon.lower())

        # Hashtags
        for raw_tag, norm_tag in zip(row["hashtags_list_raw"], row["hashtags_list_norm"]):
            entity_type = "HASHTAG_ENTITY" if _looks_like_entity(norm_tag) else "HASHTAG_TOPIC"
            surface = str(raw_tag)
            canon = _canon(norm_tag)
            reject, reason = _reject(canon, entity_type, noise)
            if reject:
                reject_reasons[video_id][reason] += 1
                continue
            mentions_rows.append(
                {
                    "run_id": run_id,
                    "video_id": video_id,
                    "entity_surface": surface,
                    "entity_canon": canon.lower(),
                    "entity_type": entity_type,
                    "source": "HASHTAG_PARSE",
                    "confidence": 0.75 if entity_type == "HASHTAG_ENTITY" else 0.55,
                    "evidence_snippet": surface,
                }
            )
            video_entities[video_id][entity_type].append(canon.lower())

        # Desc model
        desc_plain = row.get("desc_only_clean") or ""
        for surface, ent_type, conf in _desc_model_entities(desc_plain, nlp):
            canon = _canon(surface)
            reject, reason = _reject(canon, ent_type, noise)
            if reject:
                reject_reasons[video_id][reason] += 1
                continue
            mentions_rows.append(
                {
                    "run_id": run_id,
                    "video_id": video_id,
                    "entity_surface": surface,
                    "entity_canon": canon.lower(),
                    "entity_type": ent_type,
                    "source": "DESC_MODEL",
                    "confidence": conf,
                    "evidence_snippet": surface,
                }
            )
            video_entities[video_id][ent_type].append(canon.lower())

        # Structured QA placeholders
        for field in ["music_title", "poi_name"]:
            if field in row and row[field]:
                canon = _canon(str(row[field]))
                reject, reason = _reject(canon, "WORK", noise)
                if reject:
                    reject_reasons[video_id][reason] += 1
                    continue
                mentions_rows.append(
                    {
                        "run_id": run_id,
                        "video_id": video_id,
                        "entity_surface": row[field],
                        "entity_canon": canon.lower(),
                        "entity_type": "WORK",
                        "source": "QA_STRUCTURED",
                        "confidence": 0.6,
                        "evidence_snippet": str(row[field]),
                    }
                )
                video_entities[video_id]["WORK"].append(canon.lower())

    mentions_df = pd.DataFrame(mentions_rows)
    mentions_path = config.output_path("sys8_entities_mentions.parquet")
    mentions_df.to_parquet(mentions_path, index=False)

    aggregated_rows = []
    for video_id, typed_entities in video_entities.items():
        aggregated_rows.append(
            {
                "run_id": config.run_id,
                "video_id": video_id,
                "entities_by_type": dict(typed_entities),
                "reject_stats": dict(reject_reasons.get(video_id, {})),
            }
        )
    entities_by_video = pd.DataFrame(aggregated_rows)
    entities_path = config.output_path("sys8_entities_by_video.parquet")
    entities_by_video.to_parquet(entities_path, index=False)
    logging.info(
        "NER extracted %d mentions across %d videos", len(mentions_df), len(entities_by_video)
    )
    return mentions_df, entities_by_video


def ner_report(prepared: pd.DataFrame, mentions: pd.DataFrame, entities_by_video: pd.DataFrame, config: Sys8Config) -> Dict[str, float]:
    universe = len(prepared)
    coverage = len(entities_by_video) / universe if universe else 0
    garbage = mentions["entity_surface"].apply(lambda x: bool(PUNCT_RE.match(str(x)))).mean() if not mentions.empty else 0
    handle_mistype = mentions[mentions["entity_type"].isin(["PERSON", "ORG_BRAND"])]
    handle_mistype_rate = (
        handle_mistype["entity_surface"].str.contains("@").mean() if not handle_mistype.empty else 0
    )
    lines = [
        "# NER QA report",
        f"Run: {config.run_id}",
        f"Universe rows: {universe}",
        f"Coverage: {coverage:.3f}",
        f"Garbage rate: {garbage:.3f}",
        f"Handle mis-typing rate: {handle_mistype_rate:.3f}",
        "",
        "## Top rejected reasons",
    ]
    if not entities_by_video.empty and "reject_stats" in entities_by_video.columns:
        reject_total = Counter()
        for stats in entities_by_video["reject_stats"]:
            reject_total.update(stats or {})
        for reason, count in reject_total.most_common(10):
            lines.append(f"- {reason}: {count}")
    report_path = config.output_path("sys8_ner_qa_report.md")
    report_path.write_text("\n".join(lines))
    logging.info("Wrote NER QA report to %s", report_path)
    return {
        "coverage": coverage,
        "garbage_rate": garbage,
        "handle_mistype_rate": handle_mistype_rate,
    }
