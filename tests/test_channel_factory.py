"""Unit tests for app.channels.factory -- building a ChannelAdapter outside
a webhook request (#37)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.channels.factory import build_adapter
from app.channels.telegram import TelegramAdapter
from app.channels.whatsapp import WhatsAppAdapter


def _ctx(row):
    result = MagicMock()
    result.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _row(**overrides):
    row = MagicMock()
    row.bot_token = "tg-token"
    row.webhook_secret = "tg-secret"
    row.wa_phone_number_id = "12345"
    row._wa_access_token = None
    row._wa_app_secret = None
    for k, v in overrides.items():
        setattr(row, k, v)
    return row


@pytest.mark.asyncio
async def test_unknown_tenant_returns_none():
    with patch("app.channels.factory.AsyncSessionLocal", return_value=_ctx(None)):
        adapter = await build_adapter("ghost", "telegram")
    assert adapter is None


@pytest.mark.asyncio
async def test_unrecognized_channel_returns_none():
    with patch("app.channels.factory.AsyncSessionLocal", return_value=_ctx(_row())):
        adapter = await build_adapter("acme", "sms")
    assert adapter is None


@pytest.mark.asyncio
async def test_builds_telegram_adapter():
    with patch("app.channels.factory.AsyncSessionLocal", return_value=_ctx(_row())):
        adapter = await build_adapter("acme", "telegram")
    assert isinstance(adapter, TelegramAdapter)
    assert adapter._token == "tg-token"


@pytest.mark.asyncio
async def test_builds_whatsapp_adapter_decrypting_credentials():
    row = _row(_wa_access_token="enc-token", _wa_app_secret="enc-secret")
    with (
        patch("app.channels.factory.AsyncSessionLocal", return_value=_ctx(row)),
        patch("app.channels.factory.decrypt_value", side_effect=lambda v: f"dec-{v}"),
    ):
        adapter = await build_adapter("acme", "whatsapp")
    assert isinstance(adapter, WhatsAppAdapter)
    assert adapter._token == "dec-enc-token"


@pytest.mark.asyncio
async def test_whatsapp_decrypt_failure_degrades_to_raw_value_not_raise():
    row = _row(_wa_access_token="enc-token", _wa_app_secret="enc-secret")
    with (
        patch("app.channels.factory.AsyncSessionLocal", return_value=_ctx(row)),
        patch("app.channels.factory.decrypt_value", side_effect=RuntimeError("bad key")),
    ):
        adapter = await build_adapter("acme", "whatsapp")
    assert isinstance(adapter, WhatsAppAdapter)
    assert adapter._token == "enc-token"  # raw ciphertext, not raised
