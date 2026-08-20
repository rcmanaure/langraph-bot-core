import logging

from sqlalchemy import text

from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def record_denial(tenant_slug: str, thread_id: str, query: str, max_similarity: float | None) -> None:
    """Writes a closed-world not-offered denial as its own outcome
    (#51/ADR-010) — distinct from an escalation or a normal answered turn,
    and the mechanism for discovering synonym gaps (ADR-010's
    Consequences: "the log is what replaces that person"). A write failure
    must not break the reply that already went out, so it's logged and
    swallowed rather than raised."""
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text(
                    "INSERT INTO not_offered_denials (tenant_id, thread_id, query, max_similarity) "
                    "SELECT id, :thread_id, :query, :max_similarity FROM tenants WHERE slug = :slug"
                ),
                {"slug": tenant_slug, "thread_id": thread_id, "query": query, "max_similarity": max_similarity},
            )
            await db.commit()
    except Exception as exc:
        logger.warning("not_offered_denial_audit_write_failed tenant=%s err=%s", tenant_slug, exc)
