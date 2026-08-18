"""Scheduler tests. Most are mocked-DB unit tests; the expiry predicate
(claimed vs. unclaimed) is also verified against a real database in
tests/test_scheduler_live.py -- a mocked assertion on the SQL string proves
the query mentions the right table, not that the predicate is correct
(#39)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.messages import HUMAN_CONTROL_EXPIRED


def _row(thread_id="tenant:t:user:1:channel:telegram", channel="telegram", chat_id="999", user_id="1", tenant_slug="t"):
    return SimpleNamespace(thread_id=thread_id, channel=channel, chat_id=chat_id, user_id=user_id, tenant_slug=tenant_slug)


def _mock_db(rows):
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)
    return mock_db


@pytest.mark.asyncio
async def test_expire_old_interrupts_closes_the_row_after_a_successful_resume():
    """The audit row is only closed AFTER _auto_resume_and_notify confirms
    it's safe -- never unconditionally, and never before (#39)."""
    mock_db = _mock_db([_row(thread_id="t1")])

    with (
        patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)),
        patch("app.scheduler._auto_resume_and_notify", AsyncMock(return_value=True)),
        patch("app.scheduler.human_control.end", new_callable=AsyncMock) as end,
    ):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()

    mock_db.execute.assert_called_once()
    sql_clause = str(mock_db.execute.call_args[0][0])
    assert "expired_at IS NULL" in sql_clause  # a SELECT, not the UPDATE itself
    end.assert_awaited_once_with("t1")


@pytest.mark.asyncio
async def test_expire_old_interrupts_leaves_the_row_open_when_resume_fails():
    """A resume that raises (or a missing graph reference) must NOT close
    the audit row -- committing "resolved" while the checkpoint is still
    genuinely stuck at interrupt_node would strand the user silently."""
    mock_db = _mock_db([_row(thread_id="t1")])

    with (
        patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)),
        patch("app.scheduler._auto_resume_and_notify", AsyncMock(side_effect=RuntimeError("boom"))),
        patch("app.scheduler.human_control.end", new_callable=AsyncMock) as end,
    ):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()  # must not raise

    end.assert_not_awaited()


@pytest.mark.asyncio
async def test_expire_old_interrupts_excludes_claimed_threads():
    """A thread with at least one operator message SINCE this escalation
    opened never expires on the timer -- the first operator reply stops the
    clock (#39). Scoped to interrupt_started_at (not bare thread_id), since
    the same thread can escalate again long after a prior, already-closed
    escalation an operator answered."""
    mock_db = _mock_db([])

    with patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()

    sql_clause = str(mock_db.execute.call_args[0][0])
    assert "human_control_messages" in sql_clause
    assert "operator" in sql_clause
    assert "hcm.created_at >= ca.interrupt_started_at" in sql_clause


@pytest.mark.asyncio
async def test_expire_old_interrupts_no_expired():
    """When no rows are returned, nothing else runs."""
    mock_db = _mock_db([])

    with patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()  # should not raise

    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_expire_old_interrupts_one_failure_does_not_block_others():
    """A single thread's auto-resume blowing up must not stop the rest from
    being processed -- each row is independent."""
    mock_db = _mock_db([_row(thread_id="t1"), _row(thread_id="t2")])

    with (
        patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)),
        patch("app.scheduler._auto_resume_and_notify", AsyncMock(side_effect=[RuntimeError("boom"), True])) as notify,
        patch("app.scheduler.human_control.end", new_callable=AsyncMock) as end,
    ):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()  # must not raise

    assert notify.await_count == 2
    end.assert_awaited_once_with("t2")  # only the one that actually succeeded


# ── _auto_resume_and_notify -- unblocks a suspended interrupt, tells the user ──

@pytest.mark.asyncio
async def test_auto_resume_reports_not_safe_to_close_when_no_graph_wired_in():
    with patch("app.scheduler._graph", None):
        from app.scheduler import _auto_resume_and_notify
        assert await _auto_resume_and_notify(_row()) is False


@pytest.mark.asyncio
async def test_auto_resume_skips_a_proactive_escalation_never_suspended():
    """generate.py's own escalation never called interrupt() -- graph.next is
    empty, so there's nothing to resume and nobody left hanging on a reply.
    Still safe to close -- there was never anything pending."""
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=()))

    with patch("app.scheduler._graph", graph):
        from app.scheduler import _auto_resume_and_notify
        assert await _auto_resume_and_notify(_row()) is True

    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_resume_unblocks_and_notifies_the_user():
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=("interrupt_node",)))
    adapter = AsyncMock()

    with (
        patch("app.scheduler._graph", graph),
        patch("app.scheduler.build_adapter", AsyncMock(return_value=adapter)) as build,
    ):
        from app.scheduler import _auto_resume_and_notify
        assert await _auto_resume_and_notify(_row(channel="telegram", chat_id="999", tenant_slug="acme")) is True

    graph.ainvoke.assert_awaited_once()
    build.assert_awaited_once_with("acme", "telegram")
    sent_inbound, sent_text = adapter.send.await_args.args
    assert sent_inbound.chat_id == "999"
    assert sent_text == HUMAN_CONTROL_EXPIRED


@pytest.mark.asyncio
async def test_auto_resume_unblocks_but_skips_notify_without_a_chat_id():
    """Escalated before chat_id existed (migration 0015) -- the graph is
    still unblocked, just nobody to reliably notify. Still safe to close --
    unblocking the graph is what matters, not whether the notice landed."""
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=("interrupt_node",)))

    with (
        patch("app.scheduler._graph", graph),
        patch("app.scheduler.build_adapter", AsyncMock()) as build,
    ):
        from app.scheduler import _auto_resume_and_notify
        assert await _auto_resume_and_notify(_row(chat_id=None)) is True

    graph.ainvoke.assert_awaited_once()
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_resume_still_closes_when_only_the_notify_step_fails():
    """The graph is already unblocked by the time notification is attempted
    -- that's the part that must not be silently lost. A failure sending the
    fallback message (adapter error, unknown tenant) must not undo that."""
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=("interrupt_node",)))

    with (
        patch("app.scheduler._graph", graph),
        patch("app.scheduler.build_adapter", AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        from app.scheduler import _auto_resume_and_notify
        assert await _auto_resume_and_notify(_row()) is True

    graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_old_conversation_audit_deletes_and_commits():
    """purge_old_conversation_audit issues a DELETE against created_at for
    both conversation_audit and human_control_messages, and commits once."""
    mock_result = MagicMock(rowcount=3)

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)):
        from app.scheduler import purge_old_conversation_audit
        await purge_old_conversation_audit()

    assert mock_db.execute.call_count == 2
    mock_db.commit.assert_called_once()
    sql_clauses = [str(call.args[0]) for call in mock_db.execute.call_args_list]
    assert any("DELETE" in c and "conversation_audit" in c for c in sql_clauses)
    assert any("DELETE" in c and "human_control_messages" in c for c in sql_clauses)
    assert all("created_at" in c for c in sql_clauses)


@pytest.mark.asyncio
async def test_purge_old_conversation_audit_cutoff_is_ninety_days():
    mock_result = MagicMock(rowcount=0)
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)):
        from app.scheduler import _AUDIT_RETENTION_DAYS, purge_old_conversation_audit
        await purge_old_conversation_audit()

    assert _AUDIT_RETENTION_DAYS == 90
    params = mock_db.execute.call_args[0][1]
    assert "cutoff" in params


