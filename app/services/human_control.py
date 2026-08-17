"""Whether a thread is currently held by an operator, and recording what
arrives while it is. See docs/adr/ADR-009-human-control.md."""
import logging

from sqlalchemy import text

from app.channels.base import thread_id_for
from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def is_under_human_control(tenant_slug: str, channel: str, user_id: str) -> bool:
    """Whether this thread has an open escalation an operator hasn't ended.

    Reuses interrupt_node's own audit row (see app/graph/nodes/interrupt.py)
    -- an escalation is "open" exactly when that row has no expired_at yet.

    Fails closed to False: a DB hiccup degrades to the bot replying as
    normal rather than to silence, since silence with nobody watching would
    strand the user.
    """
    thread_id = thread_id_for(tenant_slug, user_id, channel)
    try:
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("""
                    SELECT 1 FROM conversation_audit
                     WHERE thread_id = :thread
                       AND expired_at IS NULL
                       AND interrupt_started_at IS NOT NULL
                """),
                {"thread": thread_id},
            )).first()
        return row is not None
    except Exception as exc:
        logger.warning("human_control_check_failed thread=%s err=%s", thread_id, exc)
        return False


async def record_message(tenant_slug: str, thread_id: str, sender: str, content: str) -> None:
    """Append to the ordered log the operator inbox reads from.

    Best-effort: a logging failure here must not turn into a second reply to
    the user, and the caller (app/channels/turn.py) has already decided not
    to send anything back.
    """
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                text("""
                    INSERT INTO human_control_messages
                        (tenant_id, thread_id, sender, content)
                    SELECT t.id, :thread, :sender, :content
                      FROM tenants t WHERE t.slug = :slug
                """),
                {
                    "thread": thread_id,
                    "sender": sender,
                    "content": content,
                    "slug": tenant_slug,
                },
            )
            await db.commit()
            # INSERT...SELECT against an unmatched slug silently inserts zero
            # rows -- no exception to catch, so the miss has to be checked
            # for explicitly or the message is lost with no trace anywhere.
            if result.rowcount == 0:
                logger.warning("human_control_record_no_tenant thread=%s slug=%s", thread_id, tenant_slug)
    except Exception as exc:
        logger.warning("human_control_record_failed thread=%s err=%s", thread_id, exc)
