"""Shared lowercase + accent-strip normalization.

One copy so vision.py's extraction-consensus comparison, md_catalog.py's
slug/keyword extraction, and any future keyword matcher (canned answers,
closed-world catalog lookup) all treat accents/case the same way instead of
drifting apart across copies.
"""

import unicodedata


def normalize_for_comparison(text_value: str) -> str:
    normalized = unicodedata.normalize("NFKD", text_value.strip().lower())
    return "".join(c for c in normalized if not unicodedata.combining(c))
