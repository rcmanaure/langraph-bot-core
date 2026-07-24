import logging

from sqlalchemy import text

from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def get_tenant_specialization(slug: str) -> str:
    """Small independent lookup shared by channel handlers (telegram.py,
    whatsapp.py) that need a tenant's specialization_context before calling
    vision extraction. generate.py does NOT use this — it already loads
    specialization_context as part of its own combined tenant SELECT.

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
