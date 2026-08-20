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


async def get_tenant_closed_world_context(slug: str) -> dict:
    """Second small lookup retrieve.py makes alongside get_tenant_specialization
    above -- expertise_area (the expansion-grade fallback text, #49/ADR-010)
    and catalog_is_closed (gates the closed-world verdict entirely). Kept
    separate rather than folded into get_tenant_specialization's query so
    that function's contract (and the ~18 existing tests patching it at
    retrieve.py's import site) stays untouched; the cost is one extra
    single-row indexed SELECT on a path that already makes one, not the
    "zero extra queries" the spec's ideal describes."""
    try:
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("SELECT expertise_area, catalog_is_closed FROM tenants WHERE slug = :s"),
                {"s": slug},
            )).first()
        if not row:
            return {"expertise_area": "", "catalog_is_closed": False}
        return {
            "expertise_area": row.expertise_area or "",
            "catalog_is_closed": bool(row.catalog_is_closed),
        }
    except Exception as exc:
        logger.warning("tenant_closed_world_context_lookup_failed slug=%s err=%s", slug, exc)
        return {"expertise_area": "", "catalog_is_closed": False}


async def get_tenant_vertical(slug: str) -> str | None:
    """Selects the tenant's prompt pack (app/services/prompt_pack.py) — see
    CONTEXT.md's "Vertical" and ADR-011. Never raises, same rationale as
    get_tenant_specialization above: a DB hiccup degrades to "no pack",
    never breaks triage."""
    try:
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("SELECT vertical FROM tenants WHERE slug = :s"),
                {"s": slug},
            )).first()
        return row.vertical if row else None
    except Exception as exc:
        logger.warning("tenant_vertical_lookup_failed slug=%s err=%s", slug, exc)
        return None
