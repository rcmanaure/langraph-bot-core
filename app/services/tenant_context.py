import logging

from sqlalchemy import text

from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def get_tenant_specialization(slug: str) -> str:
    """Small independent lookup shared by channel handlers (telegram.py,
    whatsapp.py) that need a tenant's specialization_context before calling
    vision extraction. generate.py does NOT use this — it already loads
    specialization_context as part of its own combined tenant SELECT in
    `_load_tenant()` (app/graph/nodes/generate.py).

    Deliberate duplication, not an oversight: this function fetches only
    specialization_context (channels don't need expertise_area/tone/contact),
    avoiding a second unrelated round-trip that _load_tenant()'s broader
    SELECT would otherwise force on every photo message. Cross-reference
    note (found in /code-review): if the WHERE clause here ever needs an
    `active = true` filter or similar, check `_load_tenant()` too — the two
    queries read the same tenants row and can silently drift apart.

    Never raises: a DB hiccup here must degrade to "no specialization hint",
    not break the image-handling path that calls it.
    """
    try:
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("SELECT specialization_context FROM tenants WHERE slug = :s"),
                {"s": slug},
            )).first()
        return row.specialization_context if row else ""
    except Exception as exc:
        logger.warning("tenant_specialization_lookup_failed slug=%s err=%s", slug, exc)
        return ""
