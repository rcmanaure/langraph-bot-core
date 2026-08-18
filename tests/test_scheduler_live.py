"""Integration test against a REAL Postgres database (#39).

app.scheduler.expire_old_interrupts()'s "only unclaimed expires" predicate
is a database WHERE clause (a NOT EXISTS subquery against
human_control_messages). The rest of this test suite mocks AsyncSessionLocal
for every scheduler test, which can only prove an UPDATE ran and mentions
the right table names -- it can't prove the predicate actually selects the
right rows. This file seeds a claimed escalation and an unclaimed one past
the cutoff against a real database (via app.db.AsyncSessionLocal -- the same
connection expire_old_interrupts() itself uses, set by the DATABASE_URL
environment variable), runs the job, and asserts only the unclaimed one
expired.

Opt-in: point DATABASE_URL at a reachable Postgres before running pytest.
Skips gracefully -- same contract as
tests/test_integration_ocr_medical_data.py's file-existence skips -- when
nothing is reachable, so the rest of the suite runs fine without a live DB.
"""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

from app.db import AsyncSessionLocal

pytestmark = pytest.mark.asyncio


async def _seed_tenant(session, slug: str) -> int:
    result = await session.execute(
        text("""
            INSERT INTO tenants (slug, api_key_hash, webhook_secret, bot_token, active)
            VALUES (:slug, :hash, :secret, :token, true)
            RETURNING id
        """),
        {"slug": slug, "hash": f"hash-{slug}", "secret": f"secret-{slug}", "token": f"token-{slug}"},
    )
    return result.scalar_one()


async def _seed_escalation(session, tenant_id: int, thread_id: str, started_at: datetime) -> None:
    await session.execute(
        text("""
            INSERT INTO conversation_audit
                (id, tenant_id, thread_id, user_id, channel, interrupt_started_at, created_at)
            VALUES (:id, :tid, :thread, 'seeduser', 'telegram', :started, now())
        """),
        {"id": str(uuid.uuid4()), "tid": tenant_id, "thread": thread_id, "started": started_at},
    )


async def test_only_the_unclaimed_escalation_expires():
    from app.scheduler import expire_old_interrupts

    try:
        async with AsyncSessionLocal() as probe:
            await probe.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no reachable Postgres for the live scheduler test: {exc}")

    slug = f"live-sched-{uuid.uuid4().hex[:10]}"
    claimed_thread = f"tenant:{slug}:user:1:channel:telegram"
    unclaimed_thread = f"tenant:{slug}:user:2:channel:telegram"
    past_cutoff = datetime.now(timezone.utc) - timedelta(minutes=45)  # TTL is 30 minutes

    try:
        async with AsyncSessionLocal() as session:
            tenant_id = await _seed_tenant(session, slug)
            await _seed_escalation(session, tenant_id, claimed_thread, past_cutoff)
            await _seed_escalation(session, tenant_id, unclaimed_thread, past_cutoff)
            await session.execute(
                text("""
                    INSERT INTO human_control_messages (tenant_id, thread_id, sender, content)
                    VALUES (:tid, :thread, 'operator', 'estoy en esto')
                """),
                {"tid": tenant_id, "thread": claimed_thread},
            )
            await session.commit()

        # A graph stub reporting "nothing pending" for every thread -- these
        # seeded rows have no real LangGraph checkpoint behind them, so this
        # is the accurate simulation (same as a proactive escalation that
        # never suspended). _graph=None would report every row as "not safe
        # to close" (see scheduler.py's own fail-safe) and defeat this test.
        stub_graph = AsyncMock()
        stub_graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=()))

        with patch("app.scheduler._graph", stub_graph):
            await expire_old_interrupts()

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(
                text("SELECT thread_id, expired_at FROM conversation_audit WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )).fetchall()
        by_thread = {r.thread_id: r.expired_at for r in rows}

        assert by_thread[claimed_thread] is None, "a claimed thread must never expire on the timer"
        assert by_thread[unclaimed_thread] is not None, "an unclaimed thread past the cutoff must expire"
    finally:
        # Cleanup -- this seeds into whatever database DATABASE_URL points
        # at (typically the real dev database, not a throwaway one), so
        # leaving rows behind would accumulate garbage tenants.
        async with AsyncSessionLocal() as session:
            await session.execute(text("DELETE FROM tenants WHERE slug = :slug"), {"slug": slug})
            await session.commit()
