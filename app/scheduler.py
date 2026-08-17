import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text

from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

_INTERRUPT_TTL_MINUTES = 30  # operator must respond within 30 min
_AUDIT_RETENTION_DAYS = 90

scheduler = AsyncIOScheduler(timezone="UTC")


@scheduler.scheduled_job("interval", minutes=5, id="expire_interrupts")
async def expire_old_interrupts() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_INTERRUPT_TTL_MINUTES)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("""
                UPDATE conversation_audit
                   SET expired_at = now()
                 WHERE expired_at IS NULL
                   AND interrupt_started_at IS NOT NULL
                   AND interrupt_started_at < :cutoff
                RETURNING thread_id
            """),
            {"cutoff": cutoff},
        )
        expired = [r.thread_id for r in result.fetchall()]
        await db.commit()

    if expired:
        logger.info("interrupts_expired count=%d threads=%s", len(expired), expired[:5])


@scheduler.scheduled_job("interval", hours=24, id="purge_conversation_audit")
async def purge_old_conversation_audit() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_AUDIT_RETENTION_DAYS)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("DELETE FROM conversation_audit WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
        await db.commit()

    if result.rowcount:
        logger.info("conversation_audit_purged count=%d", result.rowcount)


def start() -> None:
    if not scheduler.running:
        scheduler.start()
        logger.info("scheduler_started")


def stop() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")
