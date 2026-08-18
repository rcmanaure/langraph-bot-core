"""Builds a ChannelAdapter for a tenant from stored credentials, without a
webhook request in hand -- the operator send path (app/routes/operator.py)
needs one outside any webhook. See ADR-009 / #37.
"""
import logging

from sqlalchemy import text

from app.channels.base import ChannelAdapter
from app.channels.telegram import TelegramAdapter
from app.channels.whatsapp import WhatsAppAdapter
from app.crypto import decrypt_value
from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def build_adapter(tenant_slug: str, channel: str) -> ChannelAdapter | None:
    """None when the tenant doesn't exist, isn't active, or the channel isn't
    recognized -- callers treat all three the same way a webhook already does."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("""
                SELECT bot_token, webhook_secret, wa_phone_number_id,
                       wa_access_token AS _wa_access_token, wa_app_secret AS _wa_app_secret
                  FROM tenants WHERE slug = :s AND active = true
            """),
            {"s": tenant_slug},
        )).first()
    if not row:
        return None

    if channel == "telegram":
        return TelegramAdapter(tenant_slug, row.bot_token, row.webhook_secret)

    if channel == "whatsapp":
        # Same decrypt-with-fallback contract as whatsapp.py's webhook route:
        # a bad/rotated encryption key must degrade (send fails downstream,
        # logged) rather than 500 the caller.
        try:
            access_token = decrypt_value(row._wa_access_token) if row._wa_access_token else None
            app_secret = decrypt_value(row._wa_app_secret) if row._wa_app_secret else None
        except Exception as exc:
            logger.error("wa_decrypt_failed tenant=%s err=%s", tenant_slug, exc)
            access_token = row._wa_access_token
            app_secret = row._wa_app_secret
        return WhatsAppAdapter(tenant_slug, row.wa_phone_number_id, access_token, app_secret)

    return None
