import asyncio
import hmac
import logging
import re
import time
from dataclasses import replace

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from sqlalchemy import text

from app.channels.base import Inbound, MediaRef, SeenKeys
from app.channels.turn import run_turn
from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["telegram"])

# update_id is sequential PER BOT, not globally unique — two tenants can emit
# the same update_id, so the key must include tenant_slug (see
# TelegramAdapter.dedup_key) or one tenant's message gets silently dropped as
# a "duplicate" of another tenant's.
_SEEN = SeenKeys()


async def set_webhook(token: str, webhook_url: str, secret: str) -> bool:
    """Register a webhook with Telegram. Returns True on success."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"https://api.telegram.org/bot{token}/setWebhook",
                json={"url": webhook_url, "secret_token": secret[:256]},
            )
        data = r.json()
        ok = r.status_code == 200 and data.get("ok", False)
        if not ok:
            logger.warning("tg_set_webhook_failed token=...%s url=%s err=%s",
                           token[-6:], webhook_url, data.get("description"))
        return ok
    except Exception as exc:
        logger.warning("tg_set_webhook_error: %s", exc)
        return False


async def delete_webhook(token: str) -> None:
    """Unregister the Telegram webhook. Best-effort — errors are logged, not raised."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"https://api.telegram.org/bot{token}/deleteWebhook")
        if r.status_code != 200 or not r.json().get("ok"):
            logger.warning("tg_delete_webhook_failed token=...%s", token[-6:])
    except Exception as exc:
        logger.warning("tg_delete_webhook_error: %s", exc)


async def get_webhook_info(token: str) -> dict:
    """Fetch current webhook info from Telegram for status checks (T7)."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        data = r.json()
        if not data.get("ok"):
            return {"ok": False, "error": data.get("description", "unknown")}
        return {"ok": True, "result": data.get("result", {})}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _wa_to_tg_html(text: str) -> str:
    """Convert WhatsApp-style markdown to Telegram HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*([^\*\n]+)\*', r'<b>\1</b>', text)
    text = re.sub(r'_([^_\n]+)_', r'<i>\1</i>', text)
    return text


async def _send(token: str, chat_id: int | str, text: str) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": _wa_to_tg_html(text), "parse_mode": "HTML"},
        )
        if r.status_code != 200:
            logger.warning("tg_send_failed chat=%s status=%d body=%s", chat_id, r.status_code, r.text[:120])


async def _download_file(token: str, file_id: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as c:
        meta = await c.get(f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}")
        file_path = meta.json()["result"]["file_path"]
        return (await c.get(f"https://api.telegram.org/file/bot{token}/{file_path}")).content


# Telegram distinguishes three voice-ish message types: "voice" (voice note),
# "audio" (uploaded audio file — real mime_type varies), "video_note" (round
# video message — Whisper accepts mp4). Fixed filename/mime for the first and
# last; "audio" reports its own mime_type in the payload.
_AUDIO_MSG_TYPES = ("voice", "audio", "video_note")
_FIXED_AUDIO_FORMAT = {"voice": ("voice.ogg", "audio/ogg"), "video_note": ("video_note.mp4", "video/mp4")}


def _audio_filename_and_mime(msg_type: str, media: dict) -> tuple[str, str]:
    if msg_type in _FIXED_AUDIO_FORMAT:
        return _FIXED_AUDIO_FORMAT[msg_type]
    mime = media.get("mime_type") or "audio/mpeg"
    ext = mime.split("/")[-1] if "/" in mime else "mp3"
    return f"audio.{ext}", mime


class TelegramAdapter:
    """ChannelAdapter for the Telegram Bot API."""

    channel = "telegram"

    def __init__(self, tenant_slug: str, bot_token: str, webhook_secret: str) -> None:
        self._slug = tenant_slug
        self._token = bot_token
        self._secret = webhook_secret

    async def verify(self, request: Request) -> bool:
        """Pre-shared secret set via setWebhook, compared timing-safely (ADR-004)."""
        header = request.headers.get("x-telegram-bot-api-secret-token", "")
        return hmac.compare_digest(header, self._secret)

    def dedup_key(self, body: dict) -> str | None:
        update_id = body.get("update_id")
        return None if update_id is None else f"{self._slug}:{update_id}"

    async def parse(self, body: dict) -> list[Inbound]:
        """One Telegram update carries at most one message, so this returns 0
        or 1 — the list is the Protocol's shape, not Telegram's."""
        msg = body.get("message") or body.get("edited_message")
        if not msg:
            return []

        common = {
            "tenant_slug": self._slug,
            "channel": self.channel,
            "user_id": str((msg.get("from") or {}).get("id", "unknown")),
            "chat_id": str(msg["chat"]["id"]),
            "message_id": str(msg.get("message_id", "")),
            "caption": msg.get("caption") or "",
        }

        audio_type = next((t for t in _AUDIO_MSG_TYPES if t in msg), None)
        if audio_type:
            media = msg[audio_type]
            filename, mime_type = _audio_filename_and_mime(audio_type, media)
            return [Inbound(**common, media=[MediaRef(
                id=media["file_id"], kind="audio",
                size_bytes=media.get("file_size", 0),
                mime_type=mime_type, filename=filename,
            )])]

        if "photo" in msg:
            photo = msg["photo"][-1]  # largest resolution
            return [Inbound(**common, media=[MediaRef(
                id=photo["file_id"], kind="image",
                size_bytes=photo.get("file_size", 0),
            )])]

        if "document" in msg:
            document = msg["document"]
            return [Inbound(**common, media=[MediaRef(
                id=document.get("file_id", ""), kind="document",
                size_bytes=document.get("file_size"),
                mime_type=document.get("mime_type"),
            )])]

        content = (msg.get("text") or "").strip()
        return [Inbound(**common, text=content)] if content else []

    async def acknowledge(self, inbound: Inbound) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(
                f"https://api.telegram.org/bot{self._token}/sendChatAction",
                json={"chat_id": inbound.chat_id, "action": "typing"},
            )

    async def fetch_media(self, ref: MediaRef) -> bytes:
        return await _download_file(self._token, ref.id)

    async def send(self, inbound: Inbound, text: str) -> None:
        await _send(self._token, inbound.chat_id, text)


# Telegram albums (multi-photo messages) arrive as separate webhook updates
# sharing the same media_group_id, with no flag marking the last one — a
# multi-page medical order sent as an album would otherwise trigger one
# disconnected turn per photo instead of a single combined query. Buffer by
# group_id and flush after a debounce window with no new arrivals.
#
# Deliberately on the Telegram side of the seam: WhatsApp has no equivalent,
# so batching is a platform detail, not part of the turn. The turn accepts a
# list of MediaRef either way — an album is just N instead of 1.
#
# Safe as in-process state: entrypoint.sh pins --workers 1 (see app/runtime.py).
_MEDIA_GROUP_DEBOUNCE = 1.5
_MEDIA_GROUPS: dict[str, dict] = {}


def _album_id(body: dict) -> str | None:
    msg = body.get("message") or body.get("edited_message") or {}
    return msg.get("media_group_id")


def _buffer_album(group_id: str, inbound: Inbound, adapter: TelegramAdapter, app_state) -> None:
    group = _MEDIA_GROUPS.get(group_id)
    if group is None:
        group = {"inbound": inbound, "refs": [], "last_seen": time.monotonic()}
        _MEDIA_GROUPS[group_id] = group
        asyncio.create_task(_flush_album(group_id, adapter, app_state))
    group["refs"].extend(inbound.media)
    if inbound.caption:
        group["inbound"] = replace(group["inbound"], caption=inbound.caption)
    group["last_seen"] = time.monotonic()


async def _flush_album(group_id: str, adapter: TelegramAdapter, app_state) -> None:
    """Wait for the group to go quiet, then run one turn over every photo."""
    while True:
        await asyncio.sleep(_MEDIA_GROUP_DEBOUNCE)
        group = _MEDIA_GROUPS.get(group_id)
        if not group:
            return
        if time.monotonic() - group["last_seen"] < _MEDIA_GROUP_DEBOUNCE:
            continue
        _MEDIA_GROUPS.pop(group_id, None)
        break

    inbound = replace(group["inbound"], media=group["refs"])
    logger.info("tg_album_flushed tenant=%s photos=%d", inbound.tenant_slug, len(inbound.media))
    await run_turn(adapter, inbound, getattr(app_state, "graph", None))


@router.post("/telegram/{tenant_slug}")
async def telegram_webhook(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    # Always return 200 fast — Telegram retries on timeout causing duplicate
    # processing. All heavy work runs in the background AFTER this returns,
    # so every rejection path below also returns 200.
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("SELECT bot_token, webhook_secret FROM tenants WHERE slug = :s AND active = true"),
            {"s": tenant_slug},
        )).first()

    if not row:
        return {"ok": True}

    adapter = TelegramAdapter(tenant_slug, row.bot_token, row.webhook_secret)
    if not await adapter.verify(request):
        logger.warning("tg_bad_secret tenant=%s", tenant_slug)
        return {"ok": True}

    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("update body is not a JSON object")
    except ValueError:
        # An unhandled exception here would 500, and Telegram retries a failed
        # delivery forever instead of dropping it like every other invalid
        # update. Covers malformed JSON and valid-but-non-object JSON (a bare
        # list or string), both of which would otherwise reach body.get().
        logger.warning("tg_malformed_body tenant=%s", tenant_slug)
        return {"ok": True}

    key = adapter.dedup_key(body)
    if key is not None and _SEEN.check_and_add(key):
        logger.info("tg_duplicate_update key=%s", key)
        return {"ok": True}

    inbounds = await adapter.parse(body)
    if not inbounds:
        return {"ok": True}

    group_id = _album_id(body)
    if group_id:
        _buffer_album(group_id, inbounds[0], adapter, request.app.state)
        return {"ok": True}

    graph = getattr(request.app.state, "graph", None)
    for inbound in inbounds:
        background_tasks.add_task(run_turn, adapter, inbound, graph)
    return {"ok": True}
