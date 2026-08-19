"""Whether a thread is currently held by an operator, moving it into that
state, and recording what arrives while it is. See
docs/adr/ADR-009-human-control.md."""
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.channels.base import thread_id_for
from app.db import AsyncSessionLocal
from app.graph.thread import parse_thread_part

logger = logging.getLogger(__name__)

# Shared by every raw-SQL predicate that needs to tell "an operator message
# belonging to THIS escalation" from one left over by a prior, already-closed
# escalation on the same thread_id (thread_id is reused across escalations --
# see thread_id_for). Interpolated as a literal SQL fragment (a fixed
# constant, not request data) into app/scheduler.py's expiry predicate,
# list_pending's "attending" lookup, and thread_messages() below -- one
# definition instead of three that can silently drift apart (#39, #40).
OPERATOR_MESSAGE_SINCE_ESCALATION_SQL = (
    "hcm.sender = 'operator' AND hcm.created_at >= ca.interrupt_started_at"
)


async def start(tenant_slug: str, thread_id: str, chat_id: str = "") -> None:
    """Open the escalation that puts a thread under human control -- the
    same conversation_audit row interrupt_node's reactive suspend writes, and
    the one is_under_human_control() below reads. Idempotent: a thread that
    already has an open row (e.g. interrupt_node re-running on a resume, or
    a redelivered webhook racing this same insert) is left alone -- so
    chat_id is captured only from the turn that actually opens the
    escalation, which is exactly the guarantee an operator reply needs (see
    ADR-009 / #37: "persisted when the thread escalates").
    """
    try:
        async with AsyncSessionLocal() as db:
            existing = (await db.execute(
                text("""
                    SELECT 1 FROM conversation_audit
                     WHERE thread_id = :thread
                       AND expired_at IS NULL
                       AND interrupt_started_at IS NOT NULL
                """),
                {"thread": thread_id},
            )).first()
            if existing:
                return

            try:
                await db.execute(
                    text("""
                        INSERT INTO conversation_audit
                            (id, tenant_id, thread_id, user_id, channel,
                             chat_id, interrupt_started_at, created_at)
                        SELECT
                            :id,
                            t.id,
                            :thread,
                            :user_id,
                            :channel,
                            :chat_id,
                            :now,
                            :now
                        FROM tenants t WHERE t.slug = :slug
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "thread": thread_id,
                        "user_id": parse_thread_part(thread_id, "user"),
                        "channel": parse_thread_part(thread_id, "channel"),
                        "chat_id": chat_id or None,
                        "now": datetime.now(timezone.utc),
                        "slug": tenant_slug,
                    },
                )
                await db.commit()
            except IntegrityError:
                # Lost the race to a concurrent insert for the same thread —
                # benign, the unique partial index guarantees only one open
                # row exists.
                await db.rollback()
    except Exception as exc:
        logger.warning("human_control_start_failed thread=%s error=%s", thread_id, exc)


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


async def record_message(
    tenant_slug: str, thread_id: str, sender: str, content: str, author: str | None = None,
) -> None:
    """Append to the ordered log the operator inbox reads from.

    Best-effort: a logging failure here must not turn into a second reply to
    the user, and the caller (app/channels/turn.py) has already decided not
    to send anything back. `author` is the operator's free-text self-declared
    name (see CONTEXT.md's Operator entry) -- always None for sender="user".
    """
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                text("""
                    INSERT INTO human_control_messages
                        (tenant_id, thread_id, sender, content, author)
                    SELECT t.id, :thread, :sender, :content, :author
                      FROM tenants t WHERE t.slug = :slug
                """),
                {
                    "thread": thread_id,
                    "sender": sender,
                    "content": content,
                    "author": author,
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


async def end(thread_id: str) -> None:
    """Return a thread to the bot -- the explicit end of human control (#39).
    Silent by design (see ADR-009): this only closes the audit row, it never
    writes anything to the graph's checkpoint or to the user. Idempotent --
    a thread with no open row (already ended, or never escalated) is a no-op."""
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("""
                UPDATE conversation_audit
                   SET expired_at = :now
                 WHERE thread_id = :thread
                   AND expired_at IS NULL
                   AND interrupt_started_at IS NOT NULL
            """),
            {"now": datetime.now(timezone.utc), "thread": thread_id},
        )
        await db.commit()


async def thread_messages(thread_id: str, since: datetime | None = None) -> list[dict]:
    """The ordered human_control_messages log for one thread -- what a user
    sent while under control, and what an operator replied. Oldest first.

    `since` scopes to one escalation's messages (pass its interrupt_started_at)
    -- thread_id is reused across escalations, so an unscoped read of a
    currently-open escalation would mix in an old, already-resolved
    exchange (#40; same scoping app/scheduler.py's expiry predicate and
    list_pending's "attending" lookup already use, via
    OPERATOR_MESSAGE_SINCE_ESCALATION_SQL). Omit only when there genuinely
    is no current escalation to scope to."""
    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            text("""
                SELECT sender, content, author, created_at
                  FROM human_control_messages
                 WHERE thread_id = :thread
                   AND (CAST(:since AS timestamptz) IS NULL OR created_at >= CAST(:since AS timestamptz))
                 ORDER BY created_at
            """),
            {"thread": thread_id, "since": since},
        )
        return [dict(r._mapping) for r in rows.fetchall()]
