import logging

from sqlalchemy import text

from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def resolve_staff(tenant_slug: str, channel: str, user_id: str) -> bool:
    """Whether (channel, user_id) is an allowlisted staff member for tenant_slug.

    Identity comes only from the channel-supplied identifier — never from
    anything in the message text (see ADR-006). An empty allowlist, an
    unknown tenant, or a DB hiccup all resolve to False: fail closed, since
    this gates a privileged status.
    """
    try:
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text(
                    "SELECT 1 FROM staff_members sm "
                    "JOIN tenants t ON t.id = sm.tenant_id "
                    "WHERE t.slug = :slug AND sm.channel = :channel AND sm.identifier = :identifier"
                ),
                {"slug": tenant_slug, "channel": channel, "identifier": user_id},
            )).first()
        return row is not None
    except Exception as exc:
        logger.warning("staff_resolution_failed tenant=%s channel=%s err=%s", tenant_slug, channel, exc)
        return False
