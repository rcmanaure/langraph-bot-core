"""TTL-cached loading of a tenant's not-offered terms.

Mirrors app/services/canned.py's structure exactly (same TTL-cache +
invalidate() shape, same rationale: operator-edited data, not static
config, so a write must reach patients without waiting out the TTL). See
CONTEXT.md's "Not-offered term" and #53. See ADR-012 for why this coexists
with (rather than replaces) ADR-010's LLM/embedding closed-world mechanism.
"""

import logging
import re
import time

from sqlalchemy import text

from app.db import AsyncSessionLocal
from app.services.text_normalize import normalize_for_comparison

logger = logging.getLogger(__name__)

_TTL_SECONDS = 90  # matches canned.py's precedent

# tenant_slug -> (cached_at, rows)
_cache: dict[str, tuple[float, list[dict]]] = {}


def invalidate(tenant_slug: str) -> None:
    """Called by the admin CRUD/upload endpoints after any write so the
    change is visible on the very next message, not after the TTL."""
    _cache.pop(tenant_slug, None)


async def get_not_offered_terms(tenant_slug: str) -> list[dict]:
    now = time.monotonic()
    cached = _cache.get(tenant_slug)
    if cached and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    # Called on every non-shortcut message via match_not_offered_term() in
    # triage.py -- never raises, same defensive shape as canned.py's
    # get_canned_answers(). A DB hiccup degrades to "no terms" (falls
    # through to the normal triage/LLM path), never breaks the turn.
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                text(
                    "SELECT nt.id, nt.keywords, nt.match_mode "
                    "FROM not_offered_terms nt JOIN tenants t ON t.id = nt.tenant_id "
                    "WHERE t.slug = :slug ORDER BY nt.id"
                ),
                {"slug": tenant_slug},
            )).fetchall()
    except Exception as exc:
        logger.warning("not_offered_terms_lookup_failed slug=%s err=%s", tenant_slug, exc)
        return []

    terms = [{"id": r.id, "keywords": r.keywords, "match_mode": r.match_mode} for r in rows]
    _cache[tenant_slug] = (now, terms)
    return terms


async def match_not_offered_term(tenant_slug: str, message: str) -> bool:
    """True if `message` matches any tenant not-offered term. Word-boundary
    match over normalized text, identical semantics to
    canned.py's match_canned_answer() -- reused deliberately, not
    reimplemented, so this never drifts from that already-proven code."""
    terms = await get_not_offered_terms(tenant_slug)
    if not terms:
        return False

    normalized_message = normalize_for_comparison(message)
    for row in terms:
        keywords = [normalize_for_comparison(k) for k in (row["keywords"] or []) if k]
        if not keywords:
            continue
        hits = (re.search(rf"\b{re.escape(kw)}\b", normalized_message) is not None for kw in keywords)
        matched = all(hits) if row["match_mode"] == "all" else any(hits)
        if matched:
            return True
    return False
