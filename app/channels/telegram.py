import asyncio
import hmac
import logging
import re
import time
from collections import OrderedDict

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from langchain_core.messages import HumanMessage
from sqlalchemy import text

from app.channels.base import ChannelEvent
from app.config import settings
from app.db import AsyncSessionLocal
from app.services.tenant_context import get_tenant_specialization
from app.services.vision import MAX_MEDIA_BYTES as MAX_VOICE_BYTES
from app.services.vision import VISION_UNCERTAIN as _VISION_UNCERTAIN
from app.services.vision import extract_procedure_query as _extract_procedure_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["telegram"])

# Dedup cache: "tenant_slug:update_id" → True. Bounded to 1000 entries (LRU).
# update_id is sequential PER BOT, not globally unique — two tenants can emit
# the same update_id, so the key must include tenant_slug or one tenant's
# message gets silently dropped as a "duplicate" of another tenant's.
_SEEN_UPDATES: OrderedDict[str, bool] = OrderedDict()
_SEEN_MAX = 1000


def _is_duplicate(tenant_slug: str, update_id: int) -> bool:
    key = f"{tenant_slug}:{update_id}"
    if key in _SEEN_UPDATES:
        return True
    _SEEN_UPDATES[key] = True
    if len(_SEEN_UPDATES) > _SEEN_MAX:
        _SEEN_UPDATES.popitem(last=False)
    return False


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


# Telegram albums (multi-photo messages) arrive as separate webhook updates
# sharing the same media_group_id, with no flag marking the last one — a
# multi-page medical order sent as an album would otherwise trigger one
# disconnected graph turn per photo instead of a single combined query.
# Buffer by group_id and flush after a debounce window with no new arrivals.
# Safe as in-process state: entrypoint.sh pins --workers 1 (see app/runtime.py).
_MEDIA_GROUP_DEBOUNCE = 1.5
_MEDIA_GROUPS: dict[str, dict] = {}


async def _reply_to_event(bot_token: str, event: ChannelEvent, app_state) -> None:
    graph = getattr(app_state, "graph", None)
    if graph is None:
        logger.error("tg_graph_not_initialized thread=%s", event.thread_id)
        try:
            await _send(bot_token, event.chat_id, "Lo siento, el servicio no está disponible. Por favor intenta de nuevo más tarde.")
        except Exception:
            pass
        return

    try:
        result = await graph.ainvoke(
            {"tenant_id": event.tenant_slug, "thread_id": event.thread_id,
             "messages": [HumanMessage(content=event.text)], "answer": ""},
            config={"configurable": {"thread_id": event.thread_id}},
        )
        response = result.get("answer") or ""
        if not response and result.get("messages"):
            response = result["messages"][-1].content
        if not response:
            response = "Lo siento, no pude generar una respuesta."
    except Exception:
        logger.exception("tg_graph_failed thread=%s", event.thread_id)
        response = "Lo siento, ocurrió un error. Por favor intenta de nuevo."

    try:
        await _send(bot_token, event.chat_id, response)
    except Exception as exc:
        logger.warning("tg_final_send_failed chat=%s type=%s err=%s",
                       event.chat_id, type(response).__name__, exc, exc_info=True)


async def _process_media_group(group_id: str, tenant_slug: str, bot_token: str, app_state) -> None:
    """Debounce window: waits for the group to go quiet, then extracts every
    buffered photo and merges the confident reads into one graph turn."""
    while True:
        await asyncio.sleep(_MEDIA_GROUP_DEBOUNCE)
        group = _MEDIA_GROUPS.get(group_id)
        if not group:
            return
        if time.monotonic() - group["last_seen"] < _MEDIA_GROUP_DEBOUNCE:
            continue
        _MEDIA_GROUPS.pop(group_id, None)
        break

    chat_id = group["chat_id"]
    user_id = group["user_id"]
    caption = group["caption"]

    # Looked up once for the whole group, not per photo — tenant doesn't
    # change within a media group, so this avoids N redundant DB queries.
    specialization = await get_tenant_specialization(tenant_slug)

    queries: list[str] = []
    for photo in group["photos"]:
        if photo.get("file_size", 0) > MAX_VOICE_BYTES:
            continue
        try:
            img_bytes = await _download_file(bot_token, photo["file_id"])
            procedure_query = await _extract_procedure_query(
                img_bytes, caption, tenant_slug=tenant_slug, specialization_context=specialization
            )
        except Exception as exc:
            logger.warning("tg_vision_group_failed user=%s err=%s", user_id, exc)
            continue
        if _VISION_UNCERTAIN not in procedure_query:
            queries.append(procedure_query)

    if not queries:
        logger.warning("tg_vision_group_uncertain tenant=%s user=%s count=%d",
                        tenant_slug, user_id, len(group["photos"]))
        await _send(
            bot_token, chat_id,
            "No pude leer con seguridad los exámenes en las imágenes. Intenta con fotos "
            "más claras: buena luz, enfocadas, y que se vea toda la hoja. O si prefieres, "
            "puedes escribirme el nombre del examen o procedimiento.",
        )
        return

    # A query that already spans multiple lines (vision.py's multi-sample
    # combined_question — one photo listing several distinct items) is
    # spliced in as-is, not re-bulleted, or its own header/sub-bullets would
    # nest inside a single outer "- " bullet (found in /code-review).
    combined_query = queries[0] if len(queries) == 1 else "\n".join(
        q if "\n" in q else f"- {q}" for q in queries
    )
    logger.warning("tg_vision_group_extracted tenant=%s count=%d", tenant_slug, len(queries))
    event = ChannelEvent(
        tenant_slug=tenant_slug, channel="telegram",
        user_id=user_id, chat_id=str(chat_id),
        text=combined_query,
        thread_id=f"tenant:{tenant_slug}:user:{user_id}:channel:telegram",
    )
    await _reply_to_event(bot_token, event, app_state)


class TelegramAdapter:
    """ChannelAdapter implementation for the Telegram Bot API."""

    channel = "telegram"

    def __init__(self, tenant_slug: str, bot_token: str, webhook_secret: str) -> None:
        self._slug = tenant_slug
        self._token = bot_token
        self._secret = webhook_secret

    def verify_secret(self, secret_header: str) -> bool:
        return hmac.compare_digest(secret_header, self._secret)

    async def normalize(self, body: dict) -> ChannelEvent | None:
        msg = body.get("message") or body.get("edited_message")
        if not msg or any(k in msg for k in _AUDIO_MSG_TYPES):
            return None
        text = msg.get("text", "").strip()
        if not text:
            return None
        chat_id = str(msg["chat"]["id"])
        user_id = str((msg.get("from") or {}).get("id", "unknown"))
        return ChannelEvent(
            tenant_slug=self._slug,
            channel=self.channel,
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            thread_id=f"tenant:{self._slug}:user:{user_id}:channel:telegram",
        )

    async def send(self, event: ChannelEvent, text: str) -> None:
        await _send(self._token, event.chat_id, text)


async def _process_update(
    tenant_slug: str,
    bot_token: str,
    webhook_secret: str,
    body: dict,
    app_state,
) -> None:
    """Heavy processing: runs AFTER 200 is returned to Telegram."""
    msg = body.get("message") or body.get("edited_message")
    if not msg:
        return

    chat_id = msg["chat"]["id"]
    user_id = str((msg.get("from") or {}).get("id", "unknown"))
    adapter = TelegramAdapter(tenant_slug, bot_token, webhook_secret)

    # Feedback immediately, before any download/STT/vision work — those can
    # take several seconds and the user should see a signal right away.
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(
                f"https://api.telegram.org/bot{bot_token}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
            )
    except Exception:
        pass

    audio_type = next((t for t in _AUDIO_MSG_TYPES if t in msg), None)
    if audio_type:
        media = msg[audio_type]
        if media.get("file_size", 0) > MAX_VOICE_BYTES:
            await _send(bot_token, chat_id, "Archivo de voz demasiado grande (máx 10MB).")
            return
        filename, mime_type = _audio_filename_and_mime(audio_type, media)
        from app.services.stt import STTNotConfiguredError, transcribe
        try:
            audio = await _download_file(bot_token, media["file_id"])
            text_content = await transcribe(audio, filename, mime_type)
        except STTNotConfiguredError:
            logger.error("tg_stt_not_configured tenant=%s user=%s", tenant_slug, user_id)
            await _send(bot_token, chat_id, "La transcripción de audio no está habilitada.")
            return
        except Exception as exc:
            logger.warning("tg_stt_failed user=%s err=%s", user_id, exc)
            await _send(bot_token, chat_id,
                        "No pude procesar tu nota de voz. ¿Puedes escribirme tu consulta?")
            return
        if not text_content:
            await _send(bot_token, chat_id,
                        "No escuché nada en el audio. ¿Puedes repetirlo o escribirme?")
            return
        event = ChannelEvent(
            tenant_slug=tenant_slug, channel="telegram",
            user_id=user_id, chat_id=str(chat_id),
            text=text_content,
            thread_id=f"tenant:{tenant_slug}:user:{user_id}:channel:telegram",
        )
    elif "photo" in msg:
        if not settings.openai_vision_model:
            await _send(bot_token, chat_id, "El análisis de imágenes no está habilitado.")
            return
        photo = msg["photo"][-1]  # largest resolution
        media_group_id = msg.get("media_group_id")
        if media_group_id:
            group = _MEDIA_GROUPS.get(media_group_id)
            if group is None:
                group = {"photos": [], "caption": "", "chat_id": chat_id, "user_id": user_id,
                          "last_seen": time.monotonic()}
                _MEDIA_GROUPS[media_group_id] = group
                asyncio.create_task(_process_media_group(media_group_id, tenant_slug, bot_token, app_state))
            group["photos"].append(photo)
            if msg.get("caption"):
                group["caption"] = msg["caption"]
            group["last_seen"] = time.monotonic()
            return
        if photo.get("file_size", 0) > MAX_VOICE_BYTES:
            await _send(bot_token, chat_id, "Imagen demasiado grande (máx 10MB).")
            return
        caption = msg.get("caption", "")
        try:
            specialization = await get_tenant_specialization(tenant_slug)
            img_bytes = await _download_file(bot_token, photo["file_id"])
            procedure_query = await _extract_procedure_query(
                img_bytes, caption, tenant_slug=tenant_slug, specialization_context=specialization
            )
        except Exception as exc:
            logger.warning("tg_vision_failed user=%s err=%s", user_id, exc)
            await _send(bot_token, chat_id, "No pude procesar la imagen. Por favor intenta de nuevo.")
            return
        if _VISION_UNCERTAIN in procedure_query:
            # Don't guess and forward an uncertain read into the RAG pipeline —
            # a wrong procedure name there looks just like a confident, correct
            # answer downstream. Ask the user to type it instead.
            logger.warning("tg_vision_uncertain tenant=%s user=%s", tenant_slug, user_id)
            await _send(
                bot_token, chat_id,
                "No pude leer con seguridad el examen en la imagen. Intenta con una foto "
                "más clara: buena luz, enfocada, y que se vea toda la hoja. O si prefieres, "
                "puedes escribirme el nombre del examen o procedimiento.",
            )
            return
        logger.warning("tg_vision_extracted tenant=%s query_len=%d", tenant_slug, len(procedure_query))
        event = ChannelEvent(
            tenant_slug=tenant_slug, channel="telegram",
            user_id=user_id, chat_id=str(chat_id),
            text=procedure_query,
            thread_id=f"tenant:{tenant_slug}:user:{user_id}:channel:telegram",
        )
    else:
        event = await adapter.normalize(body)
        if not event:
            return

    await _reply_to_event(bot_token, event, app_state)


@router.post("/telegram/{tenant_slug}")
async def telegram_webhook(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    # Always return 200 fast — Telegram retries on timeout causing duplicate processing.
    # All heavy work (LLM) runs in background AFTER this handler returns.
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("SELECT bot_token, webhook_secret FROM tenants WHERE slug = :s AND active = true"),
            {"s": tenant_slug},
        )).first()

    if not row:
        return {"ok": True}

    secret_header = request.headers.get("x-telegram-bot-api-secret-token", "")
    if not hmac.compare_digest(secret_header, row.webhook_secret):
        logger.warning("tg_bad_secret tenant=%s", tenant_slug)
        return {"ok": True}

    body = await request.json()

    update_id = body.get("update_id")
    if update_id is not None and _is_duplicate(tenant_slug, update_id):
        logger.info("tg_duplicate_update update_id=%s tenant=%s", update_id, tenant_slug)
        return {"ok": True}

    background_tasks.add_task(
        _process_update,
        tenant_slug,
        row.bot_token,
        row.webhook_secret,
        body,
        request.app.state,
    )
    return {"ok": True}
