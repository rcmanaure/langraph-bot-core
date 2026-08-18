import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text

from app.channels.base import Inbound
from app.channels.factory import build_adapter
from app.db import AsyncSessionLocal
from app.messages import HUMAN_CONTROL_EXPIRED
from app.services import human_control
from app.services.human_control import OPERATOR_MESSAGE_SINCE_ESCALATION_SQL

logger = logging.getLogger(__name__)

_INTERRUPT_TTL_MINUTES = 30  # unclaimed operator must respond within 30 min
_AUDIT_RETENTION_DAYS = 90

scheduler = AsyncIOScheduler(timezone="UTC")

# Set by start(graph=...) -- the expiry job needs it to unblock a genuinely
# suspended interrupt (#39). None in any context that never wires a graph
# in (tests, a worker that only runs the purge job).
# ponytail: module-level like SeenKeys/telegram.py's _MEDIA_GROUPS -- safe
# only because entrypoint.sh pins --workers 1 (see app/runtime.py).
_graph = None


@scheduler.scheduled_job("interval", minutes=5, id="expire_interrupts")
async def expire_old_interrupts() -> None:
    """Expires only UNCLAIMED escalations -- a thread with at least one
    operator message sent SINCE this escalation opened (recorded by #37's
    send endpoint) never hits this, since the first operator reply is meant
    to stop the clock (#39). Scoped to interrupt_started_at, not just
    thread_id: thread_id is deterministic per tenant+user+channel, so the
    same thread can escalate again weeks after a prior, already-closed
    escalation an operator answered -- an unscoped match would find that old
    operator message and permanently exempt the thread from ever expiring.

    The audit row is only closed AFTER a successful resume (or after
    confirming there was nothing to resume) -- never before. Marking a row
    expired first and resuming second (the previous ordering) could commit
    "resolved" against the audit trail while the graph checkpoint was still
    genuinely stuck at interrupt_node, if the resume step then failed."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_INTERRUPT_TTL_MINUTES)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text(f"""
                SELECT ca.thread_id, ca.channel, ca.chat_id, ca.user_id, t.slug AS tenant_slug
                  FROM conversation_audit ca
                  JOIN tenants t ON t.id = ca.tenant_id
                 WHERE ca.expired_at IS NULL
                   AND ca.interrupt_started_at IS NOT NULL
                   AND ca.interrupt_started_at < :cutoff
                   AND NOT EXISTS (
                       SELECT 1 FROM human_control_messages hcm
                        WHERE hcm.thread_id = ca.thread_id
                          AND {OPERATOR_MESSAGE_SINCE_ESCALATION_SQL}
                   )
            """),
            {"cutoff": cutoff},
        )
        candidates = result.fetchall()

    if not candidates:
        return
    logger.info("interrupts_expiring count=%d threads=%s", len(candidates), [r.thread_id for r in candidates][:5])

    # ponytail: sequential, not concurrent -- each row's resume+notify is
    # its own DB/graph/channel round trip, so a tick with many candidates
    # (e.g. after an outage) takes roughly len(candidates) times as long as
    # one. Add asyncio.gather with a concurrency cap if that's ever measured
    # to matter; the 5-minute interval currently gives it plenty of slack.
    for row in candidates:
        try:
            resumed = await _auto_resume_and_notify(row)
        except Exception as exc:
            resumed = False
            logger.warning("interrupt_auto_resume_failed thread=%s err=%s", row.thread_id, exc)
        if resumed:
            await human_control.end(row.thread_id)


async def _auto_resume_and_notify(row) -> bool:
    """True once it's safe to close the audit row -- either the graph was
    genuinely unblocked, or there was never anything to unblock. False means
    "try again next tick": no graph reference to confirm state against, or
    the resume itself raised."""
    if _graph is None:
        # No graph wired in -- can't confirm what state the checkpoint is
        # in, so don't claim this escalation resolved.
        return False

    from langgraph.types import Command

    config = {"configurable": {"thread_id": row.thread_id}}
    snapshot = await _graph.aget_state(config)
    if not snapshot.next:
        return True  # proactive escalation -- never suspended, nothing to unblock

    await _graph.ainvoke(Command(resume=None), config=config)

    # Notification is best-effort from here -- the graph is already
    # unblocked, which is the part that must not be silently lost; whether
    # the user actually hears about it never gates closing the audit row.
    try:
        if not row.chat_id:
            # Escalated before chat_id existed (migration 0015) -- can't
            # reach the user reliably.
            logger.warning("interrupt_expire_notify_skipped_no_chat_id thread=%s", row.thread_id)
        else:
            adapter = await build_adapter(row.tenant_slug, row.channel)
            if adapter is not None:
                inbound = Inbound(
                    tenant_slug=row.tenant_slug, channel=row.channel, user_id=row.user_id, chat_id=row.chat_id,
                )
                await adapter.send(inbound, HUMAN_CONTROL_EXPIRED)
    except Exception as exc:
        logger.warning("interrupt_expire_notify_failed thread=%s err=%s", row.thread_id, exc)

    return True


@scheduler.scheduled_job("interval", hours=24, id="purge_conversation_audit")
async def purge_old_conversation_audit() -> None:
    """Purges both conversation_audit and human_control_messages on the same
    ninety-day cutoff -- the operator-visible message log gets the same
    retention as the turn summaries it accompanies, by the same job."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=_AUDIT_RETENTION_DAYS)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("DELETE FROM conversation_audit WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
        messages_result = await db.execute(
            text("DELETE FROM human_control_messages WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
        await db.commit()

    if result.rowcount:
        logger.info("conversation_audit_purged count=%d", result.rowcount)
    if messages_result.rowcount:
        logger.info("human_control_messages_purged count=%d", messages_result.rowcount)


def start(graph=None) -> None:
    global _graph
    _graph = graph
    if not scheduler.running:
        scheduler.start()
        logger.info("scheduler_started")


def stop() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")
