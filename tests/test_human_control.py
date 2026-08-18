"""Unit tests for app.services.human_control."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.human_control import is_under_human_control, record_message, start


def _mock_db(row=None, rowcount=1):
    result = MagicMock()
    result.first.return_value = row
    result.rowcount = rowcount
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return session, ctx


# ── start() -- opens the escalation, idempotently ───────────────────────────

@pytest.mark.asyncio
async def test_start_inserts_audit_row_when_none_open():
    """First time this thread escalates: no open row yet -> insert one."""
    session, ctx = _mock_db(row=None)
    with patch("app.services.human_control.AsyncSessionLocal", return_value=ctx):
        await start("acme", "tenant:acme:user:42:channel:telegram")

    # SELECT (existence check) + INSERT
    assert session.execute.await_count == 2
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_skips_duplicate_insert_when_already_open():
    """interrupt_node re-runs this on every resume; a second escalation
    trigger firing on an already-escalated thread must not duplicate it."""
    session, ctx = _mock_db(row=(1,))
    with patch("app.services.human_control.AsyncSessionLocal", return_value=ctx):
        await start("acme", "tenant:acme:user:42:channel:telegram")

    # Only the SELECT ran — no INSERT, no commit
    assert session.execute.await_count == 1
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_failure_never_raises():
    with patch("app.services.human_control.AsyncSessionLocal", side_effect=RuntimeError("db down")):
        await start("acme", "tenant:acme:user:42:channel:telegram")  # must not raise


@pytest.mark.asyncio
async def test_open_interrupt_row_means_under_human_control():
    session, ctx = _mock_db(row=(1,))
    with patch("app.services.human_control.AsyncSessionLocal", return_value=ctx):
        result = await is_under_human_control("acme", "telegram", "42")
    assert result is True


@pytest.mark.asyncio
async def test_no_open_interrupt_row_means_not_under_human_control():
    session, ctx = _mock_db(row=None)
    with patch("app.services.human_control.AsyncSessionLocal", return_value=ctx):
        result = await is_under_human_control("acme", "telegram", "42")
    assert result is False


@pytest.mark.asyncio
async def test_query_scopes_by_thread_id():
    session, ctx = _mock_db(row=None)
    with patch("app.services.human_control.AsyncSessionLocal", return_value=ctx):
        await is_under_human_control("acme", "whatsapp", "555")

    params = session.execute.await_args.args[1]
    assert params == {"thread": "tenant:acme:user:555:channel:whatsapp"}


@pytest.mark.asyncio
async def test_db_error_degrades_to_not_under_control_never_raises():
    """Fails closed to False -- an outage degrades to the bot replying as
    normal, not to silence with nobody watching."""
    with patch("app.services.human_control.AsyncSessionLocal", side_effect=RuntimeError("db down")):
        result = await is_under_human_control("acme", "telegram", "42")
    assert result is False


@pytest.mark.asyncio
async def test_record_message_inserts_and_commits():
    session, ctx = _mock_db()
    with patch("app.services.human_control.AsyncSessionLocal", return_value=ctx):
        await record_message("acme", "tenant:acme:user:42:channel:telegram", "user", "hola")

    session.commit.assert_awaited_once()
    params = session.execute.await_args.args[1]
    assert params == {
        "thread": "tenant:acme:user:42:channel:telegram",
        "sender": "user",
        "content": "hola",
        "slug": "acme",
    }


@pytest.mark.asyncio
async def test_record_message_unknown_tenant_logs_instead_of_failing_silently():
    """INSERT...SELECT against a slug matching no tenant row inserts zero
    rows and raises nothing -- must be checked explicitly or the message is
    lost with no trace anywhere (found in /code-review)."""
    session, ctx = _mock_db(rowcount=0)
    with (
        patch("app.services.human_control.AsyncSessionLocal", return_value=ctx),
        patch("app.services.human_control.logger") as mock_logger,
    ):
        await record_message("ghost-tenant", "tenant:ghost-tenant:user:1:channel:telegram", "user", "hola")

    session.commit.assert_awaited_once()
    mock_logger.warning.assert_called_once()
    assert "no_tenant" in mock_logger.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_record_message_failure_never_raises():
    with patch("app.services.human_control.AsyncSessionLocal", side_effect=RuntimeError("db down")):
        await record_message("acme", "tenant:acme:user:42:channel:telegram", "user", "hola")  # must not raise
