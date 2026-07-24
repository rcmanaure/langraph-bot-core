"""Unit tests for app.services.tenant_context.get_tenant_specialization."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.tenant_context import get_tenant_specialization


def _mock_db(row):
    result = MagicMock()
    result.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return patch("app.services.tenant_context.AsyncSessionLocal", return_value=ctx)


@pytest.mark.asyncio
async def test_returns_specialization_when_set():
    row = MagicMock(specialization_context="jerga médica: IGRA, antro gástrico")
    with _mock_db(row):
        result = await get_tenant_specialization("lab-tenant")
    assert result == "jerga médica: IGRA, antro gástrico"


@pytest.mark.asyncio
async def test_returns_empty_string_when_field_empty():
    row = MagicMock(specialization_context="")
    with _mock_db(row):
        result = await get_tenant_specialization("lab-tenant")
    assert result == ""


@pytest.mark.asyncio
async def test_returns_empty_string_when_tenant_not_found():
    with _mock_db(None):
        result = await get_tenant_specialization("unknown-tenant")
    assert result == ""


@pytest.mark.asyncio
async def test_never_raises_on_db_error():
    with patch("app.services.tenant_context.AsyncSessionLocal", side_effect=RuntimeError("db down")):
        result = await get_tenant_specialization("lab-tenant")
    assert result == ""
