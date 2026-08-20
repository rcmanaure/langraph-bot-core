"""TTL-cached loading of a tenant's canned answers.

Mirrors app/services/llm.py's per-factory caching intent (avoid paying a DB
round trip on every turn) but adds an explicit TTL + invalidate(), unlike
llm.py's lru_cache: canned answers are operator-edited data, not static
config, so a write must reach patients without waiting out the TTL (#47 —
that's the exact "operator fixes a wrong hours reply, patients keep getting
the old one" failure mode this feature exists to close).
"""

import time

from sqlalchemy import text

from app.db import AsyncSessionLocal

_TTL_SECONDS = 90  # matches retrieve.py's CachePolicy(ttl=90) precedent

# tenant_slug -> (cached_at, rows)
_cache: dict[str, tuple[float, list[dict]]] = {}


def invalidate(tenant_slug: str) -> None:
    """Called by the admin CRUD endpoints after any create/update/delete so
    the change is visible on the very next message, not after the TTL."""
    _cache.pop(tenant_slug, None)


async def get_canned_answers(tenant_slug: str) -> list[dict]:
    now = time.monotonic()
    cached = _cache.get(tenant_slug)
    if cached and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            text(
                "SELECT ca.id, ca.keywords, ca.match_mode, ca.answer "
                "FROM canned_answers ca JOIN tenants t ON t.id = ca.tenant_id "
                "WHERE t.slug = :slug"
            ),
            {"slug": tenant_slug},
        )).fetchall()

    answers = [
        {"id": r.id, "keywords": r.keywords, "match_mode": r.match_mode, "answer": r.answer}
        for r in rows
    ]
    _cache[tenant_slug] = (now, answers)
    return answers
