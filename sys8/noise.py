from __future__ import annotations

import collections
import logging
from typing import Dict, Iterable, List

import pandas as pd

from .config import Sys8Config


def load_noise_lists(config: Sys8Config) -> Dict[str, List[str]]:
    return {
        "stopwords": list({t.lower() for t in config.noise.get("stopwords", [])}),
        "platform": list({t.lower() for t in config.noise.get("platform", [])}),
    }


def is_noise_token(token: str, noise: Dict[str, List[str]]) -> bool:
    token = token.lower()
    return token in noise["stopwords"] or token in noise["platform"]


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = token.replace("#", " ").replace("_", " ")
    token = " ".join(token.split())
    return token


def noise_report(evidence: pd.DataFrame, config: Sys8Config) -> None:
    noise = load_noise_lists(config)
    removed_counts = collections.Counter()
    platform_tags = collections.Counter()
    kept_tokens = collections.Counter()
    for removed in evidence.get("noise_hits", []):
        for tok in removed or []:
            removed_counts[tok] += 1
            if tok in noise["platform"]:
                platform_tags[tok] += 1
    for tokens in evidence["lex_tokens"]:
        for tok in tokens:
            kept_tokens[tok] += 1
    lines = [
        "# Token noise report",
        f"Run: {config.run_id}",
        "",
        "## Top removed noise tokens",
    ]
    for tok, count in removed_counts.most_common(25):
        lines.append(f"- {tok}: {count}")
    lines.append("\n## Top platform tags encountered")
    for tok, count in platform_tags.most_common(25):
        lines.append(f"- {tok}: {count}")
    lines.append("\n## Top kept tokens")
    for tok, count in kept_tokens.most_common(25):
        lines.append(f"- {tok}: {count}")
    report_path = config.output_path("sys8_token_noise_report.md")
    report_path.write_text("\n".join(lines))
    logging.info("Wrote token noise report to %s", report_path)
