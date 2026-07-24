"""
Edge-case tests for the WhatsApp webhook handler — image (vision) and audio paths.

Mirrors tests/test_telegram_webhook.py's photo-message coverage: WhatsApp gained
image support (previously text/audio only — an image silently got a debug log
and no reply, no matter what it contained) via the shared app.services.vision
module also used by Telegram.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.channels.whatsapp import router

SLUG = "demo"
PHONE_ID = "1234567890"
ACCESS_TOKEN = "test-access-token"


# ── App factory ───────────────────────────────────────────────────────────────

def make_app(graph=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if graph is not None:
        app.state.graph = graph
    return app


# ── Payload builders ──────────────────────────────────────────────────────────

def _wrap(msg: dict) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"field": "messages", "value": {"messages": [msg]}}]}],
    }


def text_message(body="hola", msg_id="wamid.txt1", from_id="5551234"):
    return _wrap({"id": msg_id, "from": from_id, "type": "text", "text": {"body": body}})


def image_message(media_id="media1", caption="", msg_id="wamid.img1", from_id="5551234"):
    return _wrap({"id": msg_id, "from": from_id, "type": "image",
                  "image": {"id": media_id, "caption": caption}})


def audio_message(media_id="media-audio1", msg_id="wamid.audio1", from_id="5551234"):
    return _wrap({"id": msg_id, "from": from_id, "type": "audio", "audio": {"id": media_id}})


def document_message(media_id="media-doc1", filename="orden.pdf", msg_id="wamid.doc1", from_id="5551234"):
    return _wrap({"id": msg_id, "from": from_id, "type": "document",
                  "document": {"id": media_id, "filename": filename}})


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


@pytest.fixture()
def mock_db(db_row):
    """Patches AsyncSessionLocal for both the tenant lookup and the
    service-window upsert (both go through the same session)."""
    result = MagicMock()
    result.first.return_value = db_row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.channels.whatsapp.AsyncSessionLocal", return_value=ctx):
        yield


@pytest.fixture()
def mock_send():
    """Blocks the outbound WhatsApp send call and records what was sent."""
    with patch("app.channels.whatsapp._send", new_callable=AsyncMock) as m:
        yield m


@pytest.fixture(autouse=True)
def mock_typing():
    """Blocks the read-receipt/typing-indicator call for every test in this file —
    it's a real httpx POST unrelated to what most tests are checking. Tests that
    care about it (see below) reference this fixture directly to assert on it."""
    with patch("app.channels.whatsapp._mark_read_and_typing", new_callable=AsyncMock) as m:
        yield m


async def _post(app, body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(f"/webhook/whatsapp/{SLUG}", json=body)


# ── Image messages (vision extraction) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_image_extracted_query_sent_to_graph(mock_db, mock_send, mock_graph):
    """Vision extraction succeeds → extracted question becomes the graph input."""
    with (
        patch("app.channels.whatsapp.settings.openai_vision_model", "vision-model"),
        patch("app.channels.whatsapp.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.whatsapp.extract_procedure_query", new_callable=AsyncMock,
              return_value="¿Cuánto cuesta un examen de IGRA?"),
    ):
        app = make_app(mock_graph)
        r = await _post(app, image_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()
    call_input = mock_graph.ainvoke.call_args[0][0]
    assert call_input["messages"][0].content == "¿Cuánto cuesta un examen de IGRA?"


@pytest.mark.asyncio
async def test_image_uncertain_extraction_asks_user_to_type(mock_db, mock_send, mock_graph):
    """Vision model can't read the image confidently → ask the user to type it,
    never forward a guessed procedure name into the RAG pipeline."""
    from app.services.vision import VISION_UNCERTAIN

    with (
        patch("app.channels.whatsapp.settings.openai_vision_model", "vision-model"),
        patch("app.channels.whatsapp.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.whatsapp.extract_procedure_query", new_callable=AsyncMock,
              return_value=VISION_UNCERTAIN),
    ):
        app = make_app(mock_graph)
        r = await _post(app, image_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "escrib" in mock_send.call_args[0][3].lower()


@pytest.mark.asyncio
async def test_image_vision_disabled_sends_notice(mock_db, mock_send, mock_graph):
    """Vision model not configured → user-facing notice, no crash, no graph call."""
    with patch("app.channels.whatsapp.settings.openai_vision_model", ""):
        app = make_app(mock_graph)
        r = await _post(app, image_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "no está habilitado" in mock_send.call_args[0][3]


@pytest.mark.asyncio
async def test_image_extraction_failure_sends_error(mock_db, mock_send, mock_graph):
    """Vision API call raises → user gets a Spanish error, no graph call, no 500."""
    with (
        patch("app.channels.whatsapp.settings.openai_vision_model", "vision-model"),
        patch("app.channels.whatsapp.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.whatsapp.extract_procedure_query", new_callable=AsyncMock,
              side_effect=RuntimeError("Vision API returned 500")),
    ):
        app = make_app(mock_graph)
        r = await _post(app, image_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "No pude procesar la imagen" in mock_send.call_args[0][3]


@pytest.mark.asyncio
async def test_image_too_large_rejected_before_download(mock_db, mock_send, mock_graph):
    """file_size over the 10MB cap → rejected using metadata alone, full bytes never fetched."""
    with (
        patch("app.channels.whatsapp.settings.openai_vision_model", "vision-model"),
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 50 * 1024 * 1024}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock) as fetch_bytes,
        patch("app.channels.whatsapp.extract_procedure_query", new_callable=AsyncMock) as extract,
    ):
        app = make_app(mock_graph)
        r = await _post(app, image_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    fetch_bytes.assert_not_awaited()
    extract.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "grande" in mock_send.call_args[0][3].lower()


# ── Audio messages ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_transcribed_and_sent_to_graph(mock_db, mock_send, mock_graph):
    with (
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"audio"),
        patch("app.services.stt.transcribe", new_callable=AsyncMock, return_value="cuanto cuesta la biopsia"),
    ):
        app = make_app(mock_graph)
        r = await _post(app, audio_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()
    call_input = mock_graph.ainvoke.call_args[0][0]
    assert call_input["messages"][0].content == "cuanto cuesta la biopsia"


@pytest.mark.asyncio
async def test_audio_stt_failure_sends_user_error(mock_db, mock_send, mock_graph):
    """STT failure → user gets a clear Spanish error, no graph call, no silence."""
    with (
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024, "mime_type": "audio/ogg"}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"audio"),
        patch("app.services.stt.transcribe", new_callable=AsyncMock, side_effect=Exception("whisper down")),
    ):
        app = make_app(mock_graph)
        r = await _post(app, audio_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "No pude procesar tu nota de voz" in mock_send.call_args[0][3]


@pytest.mark.asyncio
async def test_audio_empty_transcription_sends_user_error(mock_db, mock_send, mock_graph):
    """Whisper returns "" (no speech detected) → user gets a clear message, not silence."""
    with (
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024, "mime_type": "audio/ogg"}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"audio"),
        patch("app.services.stt.transcribe", new_callable=AsyncMock, return_value=""),
    ):
        app = make_app(mock_graph)
        r = await _post(app, audio_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "No escuché nada" in mock_send.call_args[0][3]


@pytest.mark.asyncio
async def test_audio_stt_not_configured_sends_disabled_notice(mock_db, mock_send, mock_graph):
    """GROQ_API_KEY unset → distinct "not enabled" message, not the generic STT-failure one."""
    with (
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024, "mime_type": "audio/ogg"}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"audio"),
        patch("app.services.stt.settings.groq_api_key", ""),
    ):
        app = make_app(mock_graph)
        r = await _post(app, audio_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "no está habilitada" in mock_send.call_args[0][3]


@pytest.mark.asyncio
async def test_audio_mime_type_passed_to_transcribe(mock_db, mock_send, mock_graph):
    """Real mime_type from WhatsApp media metadata reaches Whisper — not a hardcoded
    "audio/ogg" — and any codecs param (e.g. "audio/ogg; codecs=opus") is stripped."""
    with (
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 1024,
                            "mime_type": "audio/ogg; codecs=opus"}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock, return_value=b"audio"),
        patch("app.services.stt.transcribe", new_callable=AsyncMock,
              return_value="cuanto cuesta") as mock_transcribe,
    ):
        app = make_app(mock_graph)
        r = await _post(app, audio_message())
    assert r.status_code == 200
    mock_transcribe.assert_awaited_once_with(b"audio", "audio.ogg", "audio/ogg")


@pytest.mark.asyncio
async def test_audio_too_large_rejected_before_download(mock_db, mock_send, mock_graph):
    """Same size-cap protection as images — checked via metadata before the full download."""
    with (
        patch("app.channels.whatsapp._get_media_info", new_callable=AsyncMock,
              return_value={"url": "https://example.com/media", "file_size": 50 * 1024 * 1024}),
        patch("app.channels.whatsapp._fetch_media_bytes", new_callable=AsyncMock) as fetch_bytes,
    ):
        app = make_app(mock_graph)
        r = await _post(app, audio_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    fetch_bytes.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "grande" in mock_send.call_args[0][3].lower()


# ── Document (PDF) messages ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_document_rejected_asks_for_photo(mock_db, mock_send, mock_graph):
    """PDFs aren't run through vision (no PDF→image conversion) — ask for a
    photo instead of silently dropping the message like images used to."""
    app = make_app(mock_graph)
    r = await _post(app, document_message())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    mock_send.assert_awaited_once()
    assert "PDF" in mock_send.call_args[0][3]


# ── Read receipt / typing indicator ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_typing_indicator_sent_for_every_message_type(mock_db, mock_send, mock_typing, mock_graph):
    """Every inbound message gets marked read + typing before we do anything
    else — the customer should see feedback immediately, not just once the
    graph responds several seconds later."""
    app = make_app(mock_graph)
    r = await _post(app, text_message(msg_id="wamid.typing-check"))
    assert r.status_code == 200
    mock_typing.assert_awaited_once_with(PHONE_ID, ACCESS_TOKEN, "wamid.typing-check")


# ── Unsupported types still degrade gracefully ────────────────────────────────

@pytest.mark.asyncio
async def test_unsupported_type_does_not_crash(mock_db, mock_send, mock_graph):
    msg = _wrap({"id": "wamid.sticker1", "from": "5551234", "type": "sticker", "sticker": {"id": "s1"}})
    app = make_app(mock_graph)
    r = await _post(app, msg)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
