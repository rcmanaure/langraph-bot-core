"""Scheduler tests — integration marker, needs live postgres."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_expire_old_interrupts_updates_db():
    """expire_old_interrupts issues an UPDATE and commits."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [MagicMock(thread_id="tenant:t:user:1:channel:telegram")]

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock(return_value=mock_db)

    with patch("app.scheduler.AsyncSessionLocal", mock_session):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()

    mock_db.execute.assert_called_once()
    mock_db.commit.assert_called_once()
    sql_clause = mock_db.execute.call_args[0][0]
    assert "expired_at" in str(sql_clause)


@pytest.mark.asyncio
async def test_expire_old_interrupts_no_expired():
    """When no rows are returned, no log noise."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)):
        from app.scheduler import expire_old_interrupts
        await expire_old_interrupts()  # should not raise

    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_purge_old_conversation_audit_deletes_and_commits():
    """purge_old_conversation_audit issues a DELETE against created_at and commits."""
    mock_result = MagicMock(rowcount=3)

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with patch("app.scheduler.AsyncSessionLocal", MagicMock(return_value=mock_db)):
        from app.scheduler import purge_old_conversation_audit
        await purge_old_conversation_audit()

    mock_db.execute.assert_called_once()
    mock_db.commit.assert_called_once()
    sql_clause = mock_db.execute.call_args[0][0]
    assert "DELETE" in str(sql_clause)
    assert "created_at" in str(sql_clause)


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


