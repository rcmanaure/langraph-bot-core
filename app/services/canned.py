"""TTL-cached loading of a tenant's canned answers.

Mirrors app/services/llm.py's per-factory caching intent (avoid paying a DB
round trip on every turn) but adds an explicit TTL + invalidate(), unlike
llm.py's lru_cache: canned answers are operator-edited data, not static
config, so a write must reach patients without waiting out the TTL (#47 —
that's the exact "operator fixes a wrong hours reply, patients keep getting
the old one" failure mode this feature exists to close).
"""

import logging
import re
import time

from sqlalchemy import text

from app.db import AsyncSessionLocal
from app.services.text_normalize import normalize_for_comparison

logger = logging.getLogger(__name__)

_TTL_SECONDS = 90  # matches retrieve.py's CachePolicy(ttl=90) precedent

# tenant_slug -> (cached_at, rows)
_cache: dict[str, tuple[float, list[dict]]] = {}


def invalidate(tenant_slug: str) -> None:
    """Called by the admin CRUD endpoints after any create/update/delete so
    the change is visible on the very next message, not after the TTL."""
    _cache.pop(tenant_slug, None)


async def get_canned_answers(tenant_slug: str) -> list[dict]:
    # ORDER BY id -- match_canned_answer() below returns the FIRST matching
    # row, and Postgres gives no ordering guarantee without one, which would
    # make an ambiguous two-keyword-set match nondeterministic across turns
    # (found in /code-review). Same order the admin list endpoint already uses.
    now = time.monotonic()
    cached = _cache.get(tenant_slug)
    if cached and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    # Called on every non-shortcut message via match_canned_answer() in
    # triage.py -- never raises, same defensive shape as
    # tenant_context.py's per-turn lookups. A DB hiccup degrades to "no
    # canned answers" (falls through to the normal triage/LLM path), never
    # breaks the turn.
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                text(
                    "SELECT ca.id, ca.keywords, ca.match_mode, ca.answer "
                    "FROM canned_answers ca JOIN tenants t ON t.id = ca.tenant_id "
                    "WHERE t.slug = :slug ORDER BY ca.id"
                ),
                {"slug": tenant_slug},
            )).fetchall()
    except Exception as exc:
        logger.warning("canned_answers_lookup_failed slug=%s err=%s", tenant_slug, exc)
        return []

    answers = [
        {"id": r.id, "keywords": r.keywords, "match_mode": r.match_mode, "answer": r.answer}
        for r in rows
    ]
    _cache[tenant_slug] = (now, answers)
    return answers


async def match_canned_answer(tenant_slug: str, message: str) -> str | None:
    """First tenant row whose trigger matches `message`, or None. Normalized
    at match time (app.services.text_normalize), not at write time, so a
    normalizer improvement applies retroactively without a backfill.
    Word-boundary match, not raw substring — a keyword like "hora" must
    match "a que hora abren" but not "hablar ahora" (found in /code-review);
    still matches a multi-word phrase like "clases grupales" as a unit,
    since \\b anchors on both ends of the whole keyword string."""
    answers = await get_canned_answers(tenant_slug)
    if not answers:
        return None

    normalized_message = normalize_for_comparison(message)
    for row in answers:
        keywords = [normalize_for_comparison(k) for k in (row["keywords"] or []) if k]
        if not keywords:
            continue
        hits = (re.search(rf"\b{re.escape(kw)}\b", normalized_message) is not None for kw in keywords)
        matched = all(hits) if row["match_mode"] == "all" else any(hits)
        if matched:
            return row["answer"]
    return None
