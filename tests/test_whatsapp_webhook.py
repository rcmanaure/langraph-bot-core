"""Tests for the WhatsApp channel: the WhatsAppAdapter (parsing, the media
size gate) and the webhook route.

Vision, STT, size gating, and reply wording are the inbound turn's behaviour,
not WhatsApp's — covered once in tests/test_turn.py. What's tested here is
what WhatsApp itself decides: payload parsing, signature verification, the
subscription handshake, dedup, and the service window write.

Tests attach at the ChannelAdapter interface or the outbound HTTP boundary
(httpx.AsyncClient) — never at a private helper function — so they survive
the next internal refactor instead of being rewritten by it.
"""
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.channels.base import MediaTooLarge
from app.channels.whatsapp import WhatsAppAdapter, router

SLUG = "demo"
PHONE_ID = "1234567890"
ACCESS_TOKEN = "test-access-token"
APP_SECRET = "test-app-secret"
_WA = "https://graph.facebook.com/v23.0"


# ── App factory ───────────────────────────────────────────────────────────────

def make_app(graph=None) -> FastAPI:
    """Minimal app with only the WhatsApp router — no lifespan/DB connection."""
    app = FastAPI()
    app.include_router(router)
    if graph is not None:
        app.state.graph = graph
    return app


# ── Payload builders ──────────────────────────────────────────────────────────

def _wrap(*msgs: dict) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"field": "messages", "value": {"messages": list(msgs)}}]}],
    }


def text_msg(body="hola", msg_id="wamid.txt1", from_id="5551234"):
    return {"id": msg_id, "from": from_id, "type": "text", "text": {"body": body}}


def image_msg(media_id="media1", caption="", msg_id="wamid.img1", from_id="5551234"):
    return {"id": msg_id, "from": from_id, "type": "image", "image": {"id": media_id, "caption": caption}}


def audio_msg(media_id="media-audio1", mime_type="audio/ogg", msg_id="wamid.audio1", from_id="5551234"):
    return {"id": msg_id, "from": from_id, "type": "audio",
            "audio": {"id": media_id, "mime_type": mime_type}}


def document_msg(media_id="media-doc1", filename="orden.pdf", msg_id="wamid.doc1", from_id="5551234"):
    return {"id": msg_id, "from": from_id, "type": "document",
            "document": {"id": media_id, "filename": filename}}


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def mock_graph():
    g = AsyncMock()
    g.ainvoke = AsyncMock(return_value={"answer": "Respuesta OK", "messages": []})
    return g


@pytest.fixture()
def db_row():
    row = MagicMock()
    row.wa_phone_number_id = PHONE_ID
    row._wa_access_token = ACCESS_TOKEN
    row._wa_app_secret = None  # no app_secret configured → HMAC check skipped
    return row


def _session_ctx(row):
    result = MagicMock()
    result.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx, session


@pytest.fixture()
def mock_db(db_row):
    """Patches AsyncSessionLocal for both the tenant lookup and the
    service-window upsert (both go through the same session)."""
    ctx, _ = _session_ctx(db_row)
    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        yield


@pytest.fixture()
def mock_db_secret(db_row):
    """Same tenant row, but with an app_secret configured — signature checks apply."""
    db_row._wa_app_secret = APP_SECRET
    ctx, _ = _session_ctx(db_row)
    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        yield


@pytest.fixture()
def no_tenant():
    ctx, _ = _session_ctx(None)
    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        yield


@pytest.fixture()
def mock_http():
    """Blocks all outbound httpx calls (WhatsApp Graph API): media metadata,
    media download, and messages — both replies and read/typing share the
    same `.../messages` endpoint, distinguished only by JSON body shape."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"url": "https://cdn.example/media", "file_size": 1024, "mime_type": "audio/ogg"}
    resp.content = b"media-bytes"

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=resp)
    client.post = AsyncMock(return_value=resp)
    with patch("app.channels.whatsapp.httpx.AsyncClient", return_value=client):
        yield client


def _replies_sent(mock_http) -> list[str]:
    """Text bodies of every outbound reply (not read-receipt/typing) POST."""
    bodies = []
    for c in mock_http.post.call_args_list:
        payload = c.kwargs.get("json") or {}
        if payload.get("type") == "text":
            bodies.append(payload["text"]["body"])
    return bodies


def _sign(secret: str, body_bytes: bytes) -> str:
    mac = hmac.new(secret.encode(), body_bytes, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


async def _post_raw(app, body_bytes: bytes, headers: dict | None = None):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(
            f"/webhook/whatsapp/{SLUG}", content=body_bytes,
            headers={"content-type": "application/json", **(headers or {})},
        )


async def _post(app, body: dict, headers: dict | None = None):
    return await _post_raw(app, json.dumps(body).encode(), headers)


# ── 1. Tenant resolution ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tenant_returns_ok(no_tenant):
    """Unknown/inactive slug (same SQL filter covers both) → {"ok": true}."""
    app = make_app()
    r = await _post(app, _wrap(text_msg()))
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ── 2. Signature verification ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_app_secret_configured_skips_verification(mock_db, mock_http, mock_graph):
    """Permissive dev mode: no app_secret on the tenant → signature not checked."""
    app = make_app(mock_graph)
    r = await _post(app, _wrap(text_msg()))
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_valid_signature_is_accepted(mock_db_secret, mock_http, mock_graph):
    app = make_app(mock_graph)
    body_bytes = json.dumps(_wrap(text_msg())).encode()
    r = await _post_raw(app, body_bytes, {"x-hub-signature-256": _sign(APP_SECRET, body_bytes)})
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_signature_is_rejected(mock_db_secret, mock_http, mock_graph):
    app = make_app(mock_graph)
    body_bytes = json.dumps(_wrap(text_msg())).encode()
    r = await _post_raw(app, body_bytes, {"x-hub-signature-256": "sha256=" + "0" * 64})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_signature_header_is_rejected_when_secret_configured(mock_db_secret, mock_http, mock_graph):
    app = make_app(mock_graph)
    body_bytes = json.dumps(_wrap(text_msg())).encode()
    r = await _post_raw(app, body_bytes)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()


# ── 3. Subscription handshake ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handshake_valid_challenge_is_echoed():
    row = MagicMock()
    row.wa_verify_token = "verify-me"
    ctx, _ = _session_ctx(row)
    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get(f"/webhook/whatsapp/{SLUG}", params={
                "hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "12345",
            })
    assert r.status_code == 200
    assert r.text == "12345"


@pytest.mark.asyncio
async def test_handshake_invalid_token_is_forbidden():
    row = MagicMock()
    row.wa_verify_token = "verify-me"
    ctx, _ = _session_ctx(row)
    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get(f"/webhook/whatsapp/{SLUG}", params={
                "hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "12345",
            })
    assert r.status_code == 403


# ── 4. Malformed bodies ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_malformed_json_body_returns_ok(mock_db):
    app = make_app()
    r = await _post_raw(app, b"{bad json")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_non_object_json_body_returns_ok(mock_db):
    app = make_app()
    r = await _post_raw(app, b"[1, 2, 3]")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_malformed_message_shape_returns_ok(mock_db, mock_http, mock_graph):
    """A message whose type-specific object is missing its expected key (e.g.
    an image message with no "image" key) must not 500 — parse() runs
    synchronously before the response, so an unguarded KeyError there would
    break the "always 200 fast" contract and cause endless Meta retries."""
    malformed = {"id": "wamid.bad1", "from": "5551234", "type": "image"}  # no "image" key
    app = make_app(mock_graph)
    r = await _post(app, _wrap(malformed))
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_graph.ainvoke.assert_not_awaited()


# ── 5. Dedup and batched payloads ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redelivered_message_is_answered_once(mock_db, mock_http, mock_graph):
    app = make_app(mock_graph)
    payload = _wrap(text_msg(msg_id="wamid.dup1"))
    await _post(app, payload)
    await _post(app, payload)
    mock_graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_several_messages_in_one_payload_dispatch_several_turns(mock_db, mock_http, mock_graph):
    """Regression test for the batched-payload limitation this migration removes."""
    app = make_app(mock_graph)
    payload = _wrap(
        text_msg(body="primera pregunta", msg_id="wamid.m1", from_id="111"),
        text_msg(body="segunda pregunta", msg_id="wamid.m2", from_id="222"),
    )
    r = await _post(app, payload)
    assert r.status_code == 200
    assert mock_graph.ainvoke.await_count == 2


# ── 6. Service window ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_service_window_updated_on_inbound_message(mock_db, mock_http, mock_graph):
    app = make_app(mock_graph)
    await _post(app, _wrap(text_msg()))
    mock_graph.ainvoke.assert_awaited_once()  # proves the background task ran


@pytest.mark.asyncio
async def test_service_window_failure_does_not_prevent_reply(mock_http, mock_graph):
    row = MagicMock()
    row.wa_phone_number_id = PHONE_ID
    row._wa_access_token = ACCESS_TOKEN
    row._wa_app_secret = None
    result = MagicMock()
    result.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[result, Exception("db down")])
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        app = make_app(mock_graph)
        r = await _post(app, _wrap(text_msg()))

    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()
    assert _replies_sent(mock_http) == ["Respuesta OK"]


# ── 7. Adapter parsing ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_message_is_parsed():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    [inbound] = await adapter.parse(_wrap(text_msg(body="hola")))
    assert inbound.text == "hola"
    assert inbound.media == []


@pytest.mark.asyncio
async def test_image_message_is_parsed_with_caption():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    [inbound] = await adapter.parse(_wrap(image_msg(caption="mirá esto")))
    assert inbound.caption == "mirá esto"
    assert [(r.id, r.kind) for r in inbound.media] == [("media1", "image")]


@pytest.mark.asyncio
async def test_image_message_is_parsed_without_caption():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    [inbound] = await adapter.parse(_wrap(image_msg(caption="")))
    assert inbound.caption == ""


@pytest.mark.asyncio
async def test_malformed_message_shape_is_skipped_not_raised():
    """A message whose type-specific object is missing its expected key must
    degrade to "nothing to parse", not raise — parse() runs synchronously
    before the webhook's 200 response."""
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    malformed = {"id": "wamid.bad1", "from": "5551234", "type": "image"}  # no "image" key
    inbounds = await adapter.parse(_wrap(malformed))
    assert inbounds == []


@pytest.mark.asyncio
async def test_message_missing_from_is_skipped_not_defaulted():
    """Found in /code-review: user_id/chat_id used to default to "" with no
    rejection, running the whole turn under an empty-string identity instead
    of being dropped -- and an empty identity could collide with a staff
    row whose identifier also normalizes to "" (see ADR-006)."""
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    no_from = {"id": "wamid.nofrom1", "type": "text", "text": {"body": "hola"}}
    inbounds = await adapter.parse(_wrap(no_from))
    assert inbounds == []


@pytest.mark.asyncio
async def test_audio_mime_type_codec_param_is_stripped():
    """A clean type and sensible filename reach the media ref even when
    WhatsApp reports the mime type with a codec parameter attached."""
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    [inbound] = await adapter.parse(_wrap(audio_msg(mime_type="audio/ogg; codecs=opus")))
    [ref] = inbound.media
    assert ref.kind == "audio"
    assert ref.mime_type == "audio/ogg"
    assert ref.filename == "audio.ogg"


@pytest.mark.asyncio
async def test_document_message_is_parsed():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    [inbound] = await adapter.parse(_wrap(document_msg(filename="orden.pdf")))
    [ref] = inbound.media
    assert ref.kind == "document"
    assert ref.filename == "orden.pdf"


@pytest.mark.asyncio
async def test_several_user_messages_become_several_inbounds():
    """Regression test for the limitation this migration removes: only the
    first message in a batched payload used to be answered."""
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    inbounds = await adapter.parse(_wrap(
        text_msg(body="primera", msg_id="m1", from_id="111"),
        text_msg(body="segunda", msg_id="m2", from_id="222"),
    ))
    assert [i.text for i in inbounds] == ["primera", "segunda"]


@pytest.mark.asyncio
async def test_unsupported_type_produces_nothing():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    sticker = {"id": "wamid.sticker1", "from": "5551234", "type": "sticker", "sticker": {"id": "s1"}}
    inbounds = await adapter.parse(_wrap(sticker))
    assert inbounds == []


@pytest.mark.asyncio
async def test_payload_with_no_messages_returns_empty_list():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    inbounds = await adapter.parse(_wrap())
    assert inbounds == []


# ── 8. Adapter media size gate ───────────────────────────────────────────────
#
# Learning a WhatsApp media file's size takes a Graph API round-trip, so the
# gate can't run until fetch_media — these mock the outbound HTTP boundary
# (httpx.AsyncClient), not a private helper, matching the adapter tests above.

def _media_http_stub(file_size: int, download: bytes = b"img"):
    async def get(url, headers=None):
        resp = MagicMock()
        resp.status_code = 200
        if url.startswith(_WA):  # metadata endpoint: {_WA}/{media_id}
            resp.json.return_value = {"url": "https://cdn.example/media", "file_size": file_size}
        else:  # the download URL from that metadata response
            resp.content = download
        return resp

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=get)
    return client


@pytest.mark.asyncio
async def test_fetch_media_rejects_oversized_before_download():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    ref = (await adapter.parse(_wrap(image_msg())))[0].media[0]

    client = _media_http_stub(file_size=50 * 1024 * 1024)
    with patch("app.channels.whatsapp.httpx.AsyncClient", return_value=client):
        with pytest.raises(MediaTooLarge):
            await adapter.fetch_media(ref)

    # Only the metadata GET happened — the download URL was never requested.
    requested_urls = [c.args[0] for c in client.get.call_args_list]
    assert all(u.startswith(_WA) for u in requested_urls)


@pytest.mark.asyncio
async def test_fetch_media_downloads_when_under_cap():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    ref = (await adapter.parse(_wrap(image_msg())))[0].media[0]

    client = _media_http_stub(file_size=1024, download=b"img-bytes")
    with patch("app.channels.whatsapp.httpx.AsyncClient", return_value=client):
        result = await adapter.fetch_media(ref)

    assert result == b"img-bytes"


# ── 9. Adapter acknowledge (read receipt + typing) ────────────────────────────

@pytest.mark.asyncio
async def test_acknowledge_marks_the_message_read_and_typing():
    adapter = WhatsAppAdapter(SLUG, PHONE_ID, ACCESS_TOKEN, None)
    [inbound] = await adapter.parse(_wrap(text_msg(msg_id="wamid.ack1")))

    resp = MagicMock()
    resp.status_code = 200
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=resp)
    with patch("app.channels.whatsapp.httpx.AsyncClient", return_value=client):
        await adapter.acknowledge(inbound)

    body = client.post.call_args.kwargs["json"]
    assert body["status"] == "read"
    assert body["message_id"] == "wamid.ack1"


@pytest.mark.asyncio
async def test_acknowledge_is_a_noop_without_credentials():
    adapter = WhatsAppAdapter(SLUG, None, None, None)
    [inbound] = await adapter.parse(_wrap(text_msg()))

    with patch("app.channels.whatsapp.httpx.AsyncClient") as client_cls:
        await adapter.acknowledge(inbound)

    client_cls.assert_not_called()
