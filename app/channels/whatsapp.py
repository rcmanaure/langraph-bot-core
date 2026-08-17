import hashlib
import hmac
import json
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from app.channels.base import Inbound, MediaRef, MediaTooLarge, SeenKeys
from app.channels.turn import run_turn
from app.config import MAX_MEDIA_BYTES
from app.crypto import decrypt_value
from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["whatsapp"])

# v20.0 expires 2026-09-24 (Meta guarantees ~2yr support per version) — v23.0 is
# the current stable release without v25.0's early-release risk. Bump again
# before ~2028 when v23.0 nears its own expiration.
_WA = "https://graph.facebook.com/v23.0"

# wamid is already globally unique, so unlike Telegram's per-bot update_id
# this needs no tenant prefix (see WhatsAppAdapter.dedup_key).
_SEEN = SeenKeys()


async def _send(phone_number_id: str, token: str, to: str, body: str) -> None:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{_WA}/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": to,
                  "type": "text", "text": {"body": body}},
        )
        if r.status_code != 200:
            logger.warning("wa_send_failed to=%s status=%d body=%s", to, r.status_code, r.text[:80])


async def _get_media_info(media_id: str, token: str) -> dict:
    """Fetch media metadata (url, file_size, mime_type) without downloading content —
    lets callers reject oversized media before pulling the full payload."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{_WA}/{media_id}", headers={"Authorization": f"Bearer {token}"})
        return r.json()


async def _fetch_media_bytes(url: str, token: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as c:
        return (await c.get(url, headers={"Authorization": f"Bearer {token}"})).content


async def _mark_read_and_typing(phone_number_id: str, token: str, message_id: str) -> None:
    """Blue-check the inbound message and show 'typing...' while we process it —
    vision/RAG can take several seconds and WhatsApp gives no other feedback.
    Dismissed automatically after 25s or once we reply, whichever is first.
    Best-effort: a failure here must never block the actual response."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{_WA}/{phone_number_id}/messages",
                headers={"Authorization": f"Bearer {token}"},
                json={"messaging_product": "whatsapp", "status": "read",
                      "message_id": message_id, "typing_indicator": {"type": "text"}},
            )
            if r.status_code != 200:
                logger.warning("wa_typing_indicator_failed status=%d body=%s", r.status_code, r.text[:80])
    except Exception as exc:
        logger.warning("wa_typing_indicator_error err=%s", exc)


def _clean_audio_mime(mime_type: str | None) -> tuple[str, str]:
    """WhatsApp's webhook payload reports audio mime types with a codec
    parameter attached (e.g. "audio/ogg; codecs=opus"). Strip it and derive a
    filename extension here so the turn hands the transcription service a
    clean type — that knowledge doesn't belong in the turn."""
    mime = (mime_type or "audio/ogg").split(";")[0].strip()
    ext = mime.split("/")[-1] if "/" in mime else "ogg"
    return f"audio.{ext}", mime


class WhatsAppAdapter:
    """ChannelAdapter for the WhatsApp Cloud API."""

    channel = "whatsapp"

    def __init__(
        self,
        tenant_slug: str,
        phone_number_id: str | None,
        access_token: str | None,
        app_secret: str | None,
    ) -> None:
        self._slug = tenant_slug
        self._phone_id = phone_number_id
        self._token = access_token
        self._secret = app_secret

    async def verify(self, request: Request) -> bool:
        """HMAC-SHA256 over the raw body, keyed with the tenant's app secret,
        compared timing-safely (ADR-004). No app secret configured → permissive
        (dev mode, unchanged); one configured → a missing header rejects."""
        if not self._secret:
            return True
        body_bytes = await request.body()
        sig = request.headers.get("x-hub-signature-256", "").removeprefix("sha256=")
        mac = hmac.new(self._secret.encode(), body_bytes, hashlib.sha256)
        return hmac.compare_digest(sig, mac.hexdigest())

    def dedup_key(self, body: dict) -> str | None:
        return body.get("id") or None

    async def parse(self, body: dict) -> list[Inbound]:
        """Walks the payload's entry/change/message nesting. A payload can
        carry several user messages, each becoming its own Inbound — unlike
        the single-message limitation this migration removes."""
        if body.get("object") != "whatsapp_business_account":
            return []
        inbounds: list[Inbound] = []
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") != "messages":
                    continue
                for msg in change.get("value", {}).get("messages", []):
                    inbound = self._parse_message(msg)
                    if inbound is not None:
                        inbounds.append(inbound)
        return inbounds

    def _parse_message(self, msg: dict) -> Inbound | None:
        msg_type = msg.get("type")
        from_id = msg.get("from")
        if not from_id:
            # Found in /code-review: without this guard an empty user_id/
            # chat_id ran the whole turn under an empty-string identity
            # instead of being rejected, and could collide with a staff
            # allowlist row whose identifier also normalized to "" (see
            # app/routes/admin.py's StaffMemberCreate fix, same finding).
            logger.warning("wa_malformed_message type=%s id=%s reason=missing_from", msg_type, msg.get("id"))
            return None
        common = dict(
            tenant_slug=self._slug,
            channel=self.channel,
            user_id=from_id,
            chat_id=from_id,
            message_id=msg.get("id", ""),
        )

        try:
            if msg_type == "text":
                content = (msg.get("text") or {}).get("body") or ""
                return Inbound(**common, text=content) if content else None

            if msg_type == "image":
                image = msg["image"]
                return Inbound(**common, caption=image.get("caption", ""), media=[
                    MediaRef(id=image["id"], kind="image"),
                ])

            if msg_type in ("audio", "voice"):
                audio = msg[msg_type]
                filename, mime_type = _clean_audio_mime(audio.get("mime_type"))
                return Inbound(**common, media=[
                    MediaRef(id=audio["id"], kind="audio", mime_type=mime_type, filename=filename),
                ])

            if msg_type == "document":
                document = msg["document"]
                return Inbound(**common, media=[MediaRef(
                    id=document.get("id", ""), kind="document",
                    mime_type=document.get("mime_type"), filename=document.get("filename"),
                )])
        except KeyError:
            # This runs synchronously before the webhook returns 200 (parse
            # can't be deferred — dedup needs the message ids it produces), so
            # an unhandled KeyError here would 500 the whole delivery instead
            # of degrading gracefully like every other malformed-input path.
            logger.warning("wa_malformed_message type=%s id=%s", msg_type, msg.get("id"))
            return None

        logger.debug("wa_unsupported_type type=%s from=%s", msg_type, msg.get("from"))
        return None

    async def acknowledge(self, inbound: Inbound) -> None:
        if self._phone_id and self._token and inbound.message_id:
            await _mark_read_and_typing(self._phone_id, self._token, inbound.message_id)

    async def fetch_media(self, ref: MediaRef) -> bytes:
        """Learning a WhatsApp media file's size needs this Graph API round-trip
        — it can't happen during `parse`, which must return fast so the webhook
        handler can return 200 before the platform retries. So `parse` leaves
        `size_bytes` unknown (the turn's own gate treats that as "don't gate"),
        and the cap is enforced here instead, the only place a piece of turn
        behaviour is enforced inside an adapter — see MediaTooLarge."""
        info = await _get_media_info(ref.id, self._token)
        if info.get("file_size", 0) > MAX_MEDIA_BYTES:
            raise MediaTooLarge()
        return await _fetch_media_bytes(info["url"], self._token)

    async def send(self, inbound: Inbound, text: str) -> None:
        if self._token and self._phone_id:
            await _send(self._phone_id, self._token, inbound.chat_id, text)


async def _update_service_window(tenant_slug: str, user_id: str) -> None:
    """Genuine WhatsApp-only bookkeeping (24h customer service window) with no
    Telegram equivalent — unrelated to producing a reply, so a failure here
    must not stop the turn."""
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                    INSERT INTO wa_service_windows (tenant_id, user_id, last_user_message_at)
                    SELECT t.id, :uid, now() FROM tenants t WHERE t.slug = :slug
                    ON CONFLICT (tenant_id, user_id) DO UPDATE SET last_user_message_at = now()
                """),
                {"uid": user_id, "slug": tenant_slug},
            )
            await db.commit()
    except Exception as exc:
        logger.warning("wa_service_window_update_failed err=%s", exc)


async def _run_turn_with_service_window(
    adapter: WhatsAppAdapter, inbound: Inbound, graph, tenant_slug: str,
) -> None:
    await _update_service_window(tenant_slug, inbound.user_id)
    await run_turn(adapter, inbound, graph)


@router.get("/whatsapp/{tenant_slug}")
async def whatsapp_verify(
    tenant_slug: str,
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
):
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("SELECT wa_verify_token FROM tenants WHERE slug = :s AND active = true"),
            {"s": tenant_slug},
        )).first()

    if not row or hub_mode != "subscribe" or not hmac.compare_digest(hub_verify_token or "", row.wa_verify_token or ""):
        return PlainTextResponse("Forbidden", status_code=403)
    return PlainTextResponse(hub_challenge or "")


@router.post("/whatsapp/{tenant_slug}")
async def whatsapp_webhook(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    # Always return 200 fast — Meta retries on timeout, causing duplicate
    # processing. All heavy work runs in the background AFTER this returns,
    # so every rejection path below also returns 200.
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("""
                SELECT wa_phone_number_id,
                       wa_access_token AS _wa_access_token,
                       wa_app_secret   AS _wa_app_secret
                  FROM tenants WHERE slug = :s AND active = true
            """),
            {"s": tenant_slug},
        )).first()

    if not row:
        return {"ok": True}

    try:
        access_token = decrypt_value(row._wa_access_token) if row._wa_access_token else None
        app_secret = decrypt_value(row._wa_app_secret) if row._wa_app_secret else None
    except Exception as exc:
        logger.error("wa_decrypt_failed tenant=%s err=%s", tenant_slug, exc)
        access_token = row._wa_access_token
        app_secret = row._wa_app_secret

    adapter = WhatsAppAdapter(tenant_slug, row.wa_phone_number_id, access_token, app_secret)
    if not await adapter.verify(request):
        logger.warning("wa_bad_signature tenant=%s", tenant_slug)
        return {"ok": True}

    try:
        body = json.loads(await request.body())
        if not isinstance(body, dict):
            raise ValueError("payload is not a JSON object")
    except ValueError:
        # Same contract as telegram_webhook: an unhandled exception here would
        # 500, and Meta retries a failed delivery forever instead of dropping
        # it like every other invalid payload.
        logger.warning("wa_malformed_body tenant=%s", tenant_slug)
        return {"ok": True}

    inbounds = await adapter.parse(body)
    graph = getattr(request.app.state, "graph", None)
    for inbound in inbounds:
        key = adapter.dedup_key({"id": inbound.message_id})
        if key is not None and _SEEN.check_and_add(key):
            logger.info("wa_duplicate_message key=%s", key)
            continue
        background_tasks.add_task(
            _run_turn_with_service_window, adapter, inbound, graph, tenant_slug
        )

    return {"ok": True}
