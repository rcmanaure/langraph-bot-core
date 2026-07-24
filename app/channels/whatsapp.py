import hashlib
import hmac
import json
import logging
from collections import OrderedDict

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, Query, Request
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage
from sqlalchemy import text

from app.channels.base import ChannelEvent
from app.config import settings
from app.crypto import decrypt_value
from app.db import AsyncSessionLocal
from app.services.tenant_context import get_tenant_specialization
from app.services.vision import MAX_MEDIA_BYTES, VISION_UNCERTAIN, extract_procedure_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["whatsapp"])

# v20.0 expires 2026-09-24 (Meta guarantees ~2yr support per version) — v23.0 is
# the current stable release without v25.0's early-release risk. Bump again
# before ~2028 when v23.0 nears its own expiration.
_WA = "https://graph.facebook.com/v23.0"

# Dedup cache: wamid → True. Bounded to 1000 entries (LRU).
_SEEN_WA: OrderedDict[str, bool] = OrderedDict()
_SEEN_WA_MAX = 1000


def _is_duplicate_wa(msg_id: str) -> bool:
    if msg_id in _SEEN_WA:
        return True
    _SEEN_WA[msg_id] = True
    if len(_SEEN_WA) > _SEEN_WA_MAX:
        _SEEN_WA.popitem(last=False)
    return False


class WhatsAppAdapter:
    """ChannelAdapter implementation for the WhatsApp Cloud API.

    ponytail: handler below still uses direct calls; migrate when adding a 3rd channel.
    """

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
        if not self._secret:
            return True  # no app_secret configured → allow (dev/permissive mode)
        body_bytes = await request.body()
        sig = request.headers.get("x-hub-signature-256", "").removeprefix("sha256=")
        mac = hmac.new(self._secret.encode(), body_bytes, hashlib.sha256)
        return hmac.compare_digest(sig, mac.hexdigest())

    async def normalize(self, body: dict) -> ChannelEvent | None:
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") != "messages":
                    continue
                for msg in change.get("value", {}).get("messages", []):
                    if msg.get("type") == "text":
                        from_id = msg["from"]
                        return ChannelEvent(
                            tenant_slug=self._slug,
                            channel=self.channel,
                            user_id=from_id,
                            chat_id=from_id,
                            text=msg["text"]["body"],
                            thread_id=f"tenant:{self._slug}:user:{from_id}:channel:whatsapp",
                        )
        return None

    async def send(self, event: ChannelEvent, text: str) -> None:
        if self._token and self._phone_id:
            await _send(self._phone_id, self._token, event.chat_id, text)


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
    x_hub_signature_256: str | None = Header(None),
):
    # Always return 200 fast — Meta retries on timeout, causing duplicate processing.
    # All heavy work (LLM) runs in background AFTER this handler returns.
    body_bytes = await request.body()

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

    # Decrypt secrets (no-op if FERNET_KEY not set)
    try:
        access_token = decrypt_value(row._wa_access_token) if row._wa_access_token else None
        app_secret = decrypt_value(row._wa_app_secret) if row._wa_app_secret else None
    except Exception as exc:
        logger.error("wa_decrypt_failed tenant=%s err=%s", tenant_slug, exc)
        access_token = row._wa_access_token
        app_secret = row._wa_app_secret

    # HMAC verification — when app_secret is configured, a missing header is rejected
    if app_secret:
        if not x_hub_signature_256:
            logger.warning("wa_missing_hmac tenant=%s", tenant_slug)
            return {"ok": True}
        sig = x_hub_signature_256.removeprefix("sha256=")
        mac = hmac.new(app_secret.encode(), body_bytes, hashlib.sha256)
        if not hmac.compare_digest(sig, mac.hexdigest()):
            logger.warning("wa_bad_hmac tenant=%s", tenant_slug)
            return {"ok": True}

    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        logger.warning("wa_invalid_json tenant=%s", tenant_slug)
        return {"ok": True}
    if payload.get("object") != "whatsapp_business_account":
        return {"ok": True}

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "messages":
                continue
            for msg in change.get("value", {}).get("messages", []):
                msg_id = msg.get("id", "")
                if msg_id and _is_duplicate_wa(msg_id):
                    logger.info("wa_duplicate_msg wamid=%s tenant=%s", msg_id, tenant_slug)
                    continue
                background_tasks.add_task(
                    _handle_message,
                    request=request,
                    tenant_slug=tenant_slug,
                    phone_number_id=row.wa_phone_number_id,
                    access_token=access_token,
                    msg=msg,
                )

    return {"ok": True}


async def _handle_message(
    request: Request,
    tenant_slug: str,
    phone_number_id: str | None,
    access_token: str | None,
    msg: dict,
) -> None:
    from_id = msg.get("from")
    if not from_id:
        logger.warning("wa_missing_from tenant=%s", tenant_slug)
        return
    msg_type = msg.get("type", "")
    message_id = msg.get("id", "")

    if message_id and access_token and phone_number_id:
        await _mark_read_and_typing(phone_number_id, access_token, message_id)

    # Update WhatsApp 24h service window
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                    INSERT INTO wa_service_windows (tenant_id, user_id, last_user_message_at)
                    SELECT t.id, :uid, now() FROM tenants t WHERE t.slug = :slug
                    ON CONFLICT (tenant_id, user_id) DO UPDATE SET last_user_message_at = now()
                """),
                {"uid": from_id, "slug": tenant_slug},
            )
            await db.commit()
    except Exception as exc:
        logger.warning("wa_service_window_update_failed err=%s", exc)

    if msg_type == "text":
        text_content = msg["text"]["body"]
    elif msg_type in ("audio", "voice"):
        if not access_token:
            return
        from app.services.stt import STTNotConfiguredError, transcribe
        try:
            info = await _get_media_info(msg[msg_type]["id"], access_token)
            if info.get("file_size", 0) > MAX_MEDIA_BYTES:
                if phone_number_id:
                    await _send(phone_number_id, access_token, from_id,
                                "Archivo de voz demasiado grande (máx 10MB).")
                return
            # mime_type may carry a codecs param (e.g. "audio/ogg; codecs=opus") —
            # strip it before use as both the Whisper content-type and extension.
            mime_type = (info.get("mime_type") or "audio/ogg").split(";")[0].strip()
            ext = mime_type.split("/")[-1] if "/" in mime_type else "ogg"
            audio = await _fetch_media_bytes(info["url"], access_token)
            text_content = await transcribe(audio, f"audio.{ext}", mime_type)
        except STTNotConfiguredError:
            logger.error("wa_stt_not_configured tenant=%s from=%s", tenant_slug, from_id)
            if phone_number_id:
                await _send(phone_number_id, access_token, from_id,
                            "La transcripción de audio no está habilitada.")
            return
        except Exception as exc:
            logger.warning("wa_stt_failed from=%s err=%s", from_id, exc)
            if phone_number_id:
                await _send(phone_number_id, access_token, from_id,
                            "No pude procesar tu nota de voz. ¿Puedes escribirme tu consulta?")
            return
        if not text_content:
            if phone_number_id:
                await _send(phone_number_id, access_token, from_id,
                            "No escuché nada en el audio. ¿Puedes repetirlo o escribirme?")
            return
    elif msg_type == "image":
        if not access_token or not phone_number_id:
            return
        if not settings.openai_vision_model:
            await _send(phone_number_id, access_token, from_id,
                        "El análisis de imágenes no está habilitado.")
            return
        image_obj = msg["image"]
        caption = image_obj.get("caption", "")
        try:
            info = await _get_media_info(image_obj["id"], access_token)
            if info.get("file_size", 0) > MAX_MEDIA_BYTES:
                await _send(phone_number_id, access_token, from_id,
                            "Imagen demasiado grande (máx 10MB).")
                return
            img_bytes = await _fetch_media_bytes(info["url"], access_token)
            specialization = await get_tenant_specialization(tenant_slug)
            procedure_query = await extract_procedure_query(
                img_bytes, caption, tenant_slug=tenant_slug, specialization_context=specialization
            )
        except Exception as exc:
            logger.warning("wa_vision_failed from=%s err=%s", from_id, exc)
            await _send(phone_number_id, access_token, from_id,
                        "No pude procesar la imagen. Por favor intenta de nuevo.")
            return
        if VISION_UNCERTAIN in procedure_query:
            # Don't guess and forward an uncertain read into the RAG pipeline —
            # see app/channels/telegram.py for the matching Telegram-side comment.
            logger.warning("wa_vision_uncertain tenant=%s from=%s", tenant_slug, from_id)
            await _send(
                phone_number_id, access_token, from_id,
                "No pude leer con seguridad el examen en la imagen. Intenta con una foto "
                "más clara: buena luz, enfocada, y que se vea toda la hoja. O si prefieres, "
                "puedes escribirme el nombre del examen o procedimiento.",
            )
            return
        logger.warning("wa_vision_extracted tenant=%s query=%s", tenant_slug, procedure_query[:120])
        text_content = procedure_query
    elif msg_type == "document":
        # PDF/document orders aren't run through vision (would need PDF→image
        # conversion first) — ask for a photo instead of going silent like
        # images used to before vision support was added.
        if access_token and phone_number_id:
            await _send(phone_number_id, access_token, from_id,
                        "Por ahora no puedo leer documentos PDF. ¿Puedes mandarme una foto del examen?")
        return
    else:
        logger.debug("wa_unsupported_type type=%s from=%s", msg_type, from_id)
        return

    if not text_content:
        return

    thread_id = f"tenant:{tenant_slug}:user:{from_id}:channel:whatsapp"
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        logger.error("wa_graph_not_initialized thread=%s", thread_id)
        if access_token and phone_number_id:
            await _send(phone_number_id, access_token, from_id, "Lo siento, el servicio no está disponible. Por favor intenta de nuevo más tarde.")
        return
    try:
        result = await graph.ainvoke(
            {"tenant_id": tenant_slug, "thread_id": thread_id,
             "messages": [HumanMessage(content=text_content)], "answer": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = result.get("answer") or ""
        if not response and result.get("messages"):
            response = result["messages"][-1].content
    except Exception:
        logger.exception("wa_graph_failed thread=%s", thread_id)
        response = "Lo siento, ocurrió un error."

    if response and access_token and phone_number_id:
        await _send(phone_number_id, access_token, from_id, response)
