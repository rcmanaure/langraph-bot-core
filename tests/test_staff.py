"""Unit tests for app.services.staff.resolve_staff."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.staff import resolve_staff


def _mock_db(row):
    result = MagicMock()
    result.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return session, patch("app.services.staff.AsyncSessionLocal", return_value=ctx)


@pytest.mark.asyncio
async def test_allowlisted_identity_resolves_as_staff():
    session, mock_db = _mock_db(row=(1,))
    with mock_db:
        result = await resolve_staff("acme", "telegram", "42")
    assert result is True


@pytest.mark.asyncio
async def test_non_allowlisted_identity_does_not_resolve():
    session, mock_db = _mock_db(row=None)
    with mock_db:
        result = await resolve_staff("acme", "telegram", "42")
    assert result is False


@pytest.mark.asyncio
async def test_query_scopes_by_tenant_channel_and_identifier():
    """The WHERE clause must filter on all three — a memory that only
    checked identifier would honour staff across tenants and channels."""
    session, mock_db = _mock_db(row=None)
    with mock_db:
        await resolve_staff("acme", "whatsapp", "555")

    params = session.execute.await_args.args[1]
    assert params == {"slug": "acme", "channel": "whatsapp", "identifier": "555"}


@pytest.mark.asyncio
async def test_empty_allowlist_resolves_false_not_an_error():
    session, mock_db = _mock_db(row=None)
    with mock_db:
        result = await resolve_staff("no-staff-tenant", "telegram", "1")
    assert result is False


@pytest.mark.asyncio
async def test_never_raises_on_db_error():
    with patch("app.services.staff.AsyncSessionLocal", side_effect=RuntimeError("db down")):
        result = await resolve_staff("acme", "telegram", "42")
    assert result is False
