"""
Edge-case tests for the Telegram webhook handler.
All DB and HTTP calls are mocked — no live services required.

Known bugs surfaced here (marked with BUG):
  - sendChatAction network error propagates as 500 instead of 200
  - _send() network error propagates as 500 instead of 200
  - Graph returning empty answer + empty messages → sends empty string to user
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.channels.telegram import router

SLUG = "demo"
SECRET = "test-secret-123"
BOT = "999:FAKE"


# ── App factory ───────────────────────────────────────────────────────────────

def make_app(graph=None) -> FastAPI:
    """Minimal app with only the Telegram router — no lifespan/DB connection."""
    app = FastAPI()
    app.include_router(router)
    if graph is not None:
        app.state.graph = graph
    return app


# ── Payload builders ──────────────────────────────────────────────────────────

def text_update(text="hola", chat_id=100, user_id=42):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


def voice_update(file_id="f1", file_size=1024, chat_id=100, user_id=42):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": user_id},
                        "voice": {"file_id": file_id, "file_size": file_size}}}


def photo_update(file_id="p1", caption="", chat_id=100, user_id=42):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "caption": caption,
                        "photo": [{"file_id": file_id, "file_size": 1024}]}}


def photo_group_update(file_id, media_group_id, caption="", chat_id=100, user_id=42):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "caption": caption,
                        "media_group_id": media_group_id,
                        "photo": [{"file_id": file_id, "file_size": 1024}]}}


def edited_update(text="edited text", chat_id=100, user_id=42):
    return {"edited_message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_media_groups():
    """_MEDIA_GROUPS is module-level state — must reset between tests or a
    group buffered in one test could leak into an unrelated later test."""
    from app.channels.telegram import _MEDIA_GROUPS
    _MEDIA_GROUPS.clear()
    yield
    _MEDIA_GROUPS.clear()


@pytest.fixture()
def mock_graph():
    g = AsyncMock()
    g.ainvoke = AsyncMock(return_value={"answer": "Respuesta OK", "messages": []})
    return g


@pytest.fixture()
def db_row():
    row = MagicMock()
    row.webhook_secret = SECRET
    row.bot_token = BOT
    return row


@pytest.fixture()
def mock_db(db_row):
    """Patches AsyncSessionLocal to return a valid tenant row."""
    result = MagicMock()
    result.first.return_value = db_row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.channels.telegram.AsyncSessionLocal", return_value=ctx):
        yield


@pytest.fixture()
def no_tenant():
    """Patches AsyncSessionLocal to return nothing (tenant not found)."""
    result = MagicMock()
    result.first.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.channels.telegram.AsyncSessionLocal", return_value=ctx):
        yield


@pytest.fixture()
def mock_http():
    """Blocks all outbound httpx calls (Telegram API)."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"result": {"file_path": "voice/abc.ogg"}}
    resp.content = b"fake_audio"
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock(return_value=resp)
    with patch("app.channels.telegram.httpx.AsyncClient", return_value=client):
        yield client


async def _post(app, body, secret=SECRET):
    """Helper: POST to the webhook endpoint."""
    headers = {"x-telegram-bot-api-secret-token": secret} if secret else {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(f"/webhook/telegram/{SLUG}", json=body, headers=headers)


# ── 1. Tenant resolution ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_slug_returns_ok(no_tenant):
    """Unknown slug → silently returns {"ok": true}. No error to attacker."""
    app = make_app()
    r = await _post(app, text_update())
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_inactive_tenant_returns_ok():
    """Inactive tenant (active=false filtered by SQL) → same as not found."""
    # DB returns None because query has AND active = true
    result = MagicMock()
    result.first.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.channels.telegram.AsyncSessionLocal", return_value=ctx):
        app = make_app()
        r = await _post(app, text_update())
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ── 2. Secret validation ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_secret_header_returns_ok(mock_db):
    """No secret header → {"ok": true}, not 401. Telegram retries are safe."""
    app = make_app()
    r = await _post(app, text_update(), secret=None)
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_wrong_secret_returns_ok(mock_db):
    """Wrong secret → {"ok": true}. Timing-safe via hmac.compare_digest."""
    app = make_app()
    r = await _post(app, text_update(), secret="wrong-secret")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_empty_string_secret_returns_ok(mock_db):
    """Empty string secret (not the same as missing header) → rejected."""
    app = make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            f"/webhook/telegram/{SLUG}",
            json=text_update(),
            headers={"x-telegram-bot-api-secret-token": ""},
        )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_malformed_json_body_returns_ok(mock_db):
    """Regression: QA on 2026-08-03 found a valid-secret request with a
    malformed JSON body raised an unhandled JSONDecodeError -> 500, breaking
    this handler's own "always 200 fast" contract (comment above
    telegram_webhook) and making Telegram retry the bad update forever."""
    app = make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            f"/webhook/telegram/{SLUG}",
            content=b"{bad json",
            headers={
                "x-telegram-bot-api-secret-token": SECRET,
                "content-type": "application/json",
            },
        )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_valid_json_non_object_body_returns_ok(mock_db):
    """Same contract as above for valid-but-non-dict JSON (bare list/string),
    which would otherwise reach body.get() and raise AttributeError."""
    app = make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            f"/webhook/telegram/{SLUG}",
            content=b"[1, 2, 3]",
            headers={
                "x-telegram-bot-api-secret-token": SECRET,
                "content-type": "application/json",
            },
        )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ── 3. Message parsing ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_message_key_returns_ok(mock_db, mock_http):
    """Payload with no message/edited_message (e.g. channel post) → ok."""
    app = make_app()
    r = await _post(app, {"update_id": 1, "channel_post": {"text": "hi"}})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_message_missing_chat_key_returns_ok_not_500(mock_db, mock_http, mock_graph):
    """Found in /code-review: parse() indexed msg["chat"]["id"] unguarded --
    an update missing "chat" raised KeyError uncaught, 500ing the webhook.
    Worse, dedup_key() marks the update_id seen BEFORE parse() runs, so a
    crash here meant Telegram's automatic retry of the same update_id got
    silently dropped by the dedup check instead of getting a second try."""
    app = make_app(mock_graph)
    payload = {"update_id": 99, "message": {"from": {"id": 1}, "text": "hola"}}  # no 'chat'
    r = await _post(app, payload)
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_edited_message_is_processed(mock_db, mock_http, mock_graph):
    """edited_message (no message key) should be processed normally."""
    app = make_app(mock_graph)
    r = await _post(app, edited_update("texto editado"))
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    mock_graph.ainvoke.assert_awaited_once()
    call_input = mock_graph.ainvoke.call_args[0][0]
    assert call_input["messages"][0].content == "texto editado"


@pytest.mark.asyncio
async def test_message_takes_priority_over_edited_message(mock_db, mock_http, mock_graph):
    """When both message and edited_message are present, message wins."""
    app = make_app(mock_graph)
    payload = {
        "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "original"},
        "edited_message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "edited"},
    }
    await _post(app, payload)
    call_input = mock_graph.ainvoke.call_args[0][0]
    assert call_input["messages"][0].content == "original"


@pytest.mark.asyncio
async def test_whitespace_only_text_returns_ok(mock_db, mock_http):
    """Message with only spaces strips to empty → no graph call."""
    app = make_app()
    r = await _post(app, text_update("   "))
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_empty_text_returns_ok(mock_db, mock_http):
    """Message with empty string text → no graph call."""
    app = make_app()
    r = await _post(app, text_update(""))
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_missing_from_field_uses_unknown(mock_db, mock_http, mock_graph):
    """Message without 'from' field (can happen in groups) → user_id = 'unknown'."""
    app = make_app(mock_graph)
    payload = {"message": {"chat": {"id": 1}, "text": "hola"}}  # no 'from'
    await _post(app, payload)
    call_cfg = mock_graph.ainvoke.call_args[1]["config"]
    assert ":user:unknown:" in call_cfg["configurable"]["thread_id"]


# ── 4. Thread ID format ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_thread_id_format(mock_db, mock_http, mock_graph):
    """thread_id must be: tenant:{slug}:user:{user_id}:channel:telegram"""
    app = make_app(mock_graph)
    await _post(app, text_update(user_id=99))
    call_cfg = mock_graph.ainvoke.call_args[1]["config"]
    assert call_cfg["configurable"]["thread_id"] == f"tenant:{SLUG}:user:99:channel:telegram"


# ── 5. Graph invocation & response ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_answer_sent_to_user(mock_db, mock_http, mock_graph):
    """Graph answer is forwarded to sendMessage."""
    mock_graph.ainvoke = AsyncMock(return_value={"answer": "Precio: $100", "messages": []})
    app = make_app(mock_graph)
    await _post(app, text_update())
    # Verify _send was called with the answer text
    send_calls = mock_http.post.call_args_list
    send_msg = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert send_msg is not None
    assert "Precio: $100" in str(send_msg)


@pytest.mark.asyncio
async def test_fallback_to_messages_when_answer_empty(mock_db, mock_http, mock_graph):
    """If answer is empty, last message content is used."""
    from langchain_core.messages import AIMessage
    mock_graph.ainvoke = AsyncMock(return_value={
        "answer": "",
        "messages": [AIMessage(content="fallback desde messages")],
    })
    app = make_app(mock_graph)
    await _post(app, text_update())
    send_calls = mock_http.post.call_args_list
    send_msg = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert "fallback desde messages" in str(send_msg)


@pytest.mark.asyncio
async def test_graph_exception_sends_spanish_error(mock_db, mock_http, mock_graph):
    """Graph crash → user receives Spanish apology, NOT a 500."""
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    app = make_app(mock_graph)
    r = await _post(app, text_update())
    assert r.status_code == 200
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert "Lo siento" in str(error_call)


@pytest.mark.asyncio
async def test_empty_answer_and_empty_messages_sends_fallback(mock_db, mock_http, mock_graph):
    """Graph returning empty answer + empty messages → sends a fallback message (not empty string)."""
    mock_graph.ainvoke = AsyncMock(return_value={"answer": "", "messages": []})
    app = make_app(mock_graph)
    r = await _post(app, text_update())
    assert r.status_code == 200
    send_calls = mock_http.post.call_args_list
    send_msg = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert send_msg is not None
    sent_text = send_msg[1]["json"]["text"]
    assert sent_text != ""  # Never sends empty string to Telegram
    assert "Lo siento" in sent_text


# ── 6. Voice messages ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_over_10mb_sends_size_error(mock_db, mock_http, mock_graph):
    """Voice > 10 MB → sends size error, no graph call."""
    app = make_app(mock_graph)
    payload = voice_update(file_size=11 * 1024 * 1024)
    r = await _post(app, payload)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert "10MB" in str(error_call)


@pytest.mark.asyncio
async def test_voice_exactly_at_limit_is_processed(mock_db, mock_http, mock_graph):
    """Voice at exactly 10 MB (not over) → STT proceeds."""
    with patch("app.channels.telegram._download_file", new_callable=AsyncMock,
               return_value=b"audio"), \
         patch("app.services.stt.transcribe", new_callable=AsyncMock,
               return_value="texto transcrito"):
        app = make_app(mock_graph)
        payload = voice_update(file_size=10 * 1024 * 1024)
        r = await _post(app, payload)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_stt_failure_sends_user_error(mock_db, mock_http, mock_graph):
    """STT failure → user gets a clear Spanish error, no graph call, no silence."""
    app = make_app(mock_graph)
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, return_value=b"audio"), \
         patch("app.services.stt.transcribe",
               new_callable=AsyncMock, side_effect=Exception("whisper down")):
        r = await _post(app, voice_update())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert error_call is not None
    assert "No pude procesar tu nota de voz" in str(error_call)


@pytest.mark.asyncio
async def test_voice_empty_transcription_sends_user_error(mock_db, mock_http, mock_graph):
    """Whisper returns "" (no speech detected) → user gets a clear message, not silence."""
    app = make_app(mock_graph)
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, return_value=b"audio"), \
         patch("app.services.stt.transcribe", new_callable=AsyncMock, return_value=""):
        r = await _post(app, voice_update())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert error_call is not None
    assert "No escuché nada" in str(error_call)


@pytest.mark.asyncio
async def test_voice_stt_not_configured_sends_disabled_notice(mock_db, mock_http, mock_graph):
    """GROQ_API_KEY unset → distinct "not enabled" message, not the generic STT-failure one."""
    app = make_app(mock_graph)
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, return_value=b"audio"), \
         patch("app.services.stt.settings.groq_api_key", ""):
        r = await _post(app, voice_update())
    assert r.status_code == 200
    mock_graph.ainvoke.assert_not_awaited()
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert error_call is not None
    assert "no está habilitada" in str(error_call)


@pytest.mark.asyncio
async def test_audio_message_type_is_transcribed(mock_db, mock_http, mock_graph):
    """Telegram "audio" (uploaded file, distinct from "voice") is transcribed and processed."""
    payload = {"message": {
        "chat": {"id": 100}, "from": {"id": 42},
        "audio": {"file_id": "a1", "file_size": 1024, "mime_type": "audio/mpeg"},
    }}
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, return_value=b"audio"), \
         patch("app.services.stt.transcribe", new_callable=AsyncMock,
               return_value="cuanto cuesta") as mock_transcribe:
        app = make_app(mock_graph)
        r = await _post(app, payload)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()
    mock_transcribe.assert_awaited_once_with(b"audio", "audio.mpeg", "audio/mpeg")


@pytest.mark.asyncio
async def test_video_note_message_type_is_transcribed(mock_db, mock_http, mock_graph):
    """Telegram "video_note" (round video) is transcribed via Whisper as mp4."""
    payload = {"message": {
        "chat": {"id": 100}, "from": {"id": 42},
        "video_note": {"file_id": "v1", "file_size": 1024},
    }}
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, return_value=b"video"), \
         patch("app.services.stt.transcribe", new_callable=AsyncMock,
               return_value="cuanto cuesta") as mock_transcribe:
        app = make_app(mock_graph)
        r = await _post(app, payload)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()
    mock_transcribe.assert_awaited_once_with(b"video", "video_note.mp4", "video/mp4")


@pytest.mark.asyncio
async def test_typing_sent_before_download_starts(mock_db, mock_http, mock_graph):
    """Typing indicator fires immediately, before download/STT — proven by the
    download itself failing and typing having already been sent regardless."""
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, side_effect=Exception("network down")):
        app = make_app(mock_graph)
        r = await _post(app, voice_update())
    assert r.status_code == 200
    typing_call = next((c for c in mock_http.post.call_args_list if "sendChatAction" in str(c)), None)
    assert typing_call is not None


@pytest.mark.asyncio
async def test_voice_missing_file_size_treated_as_zero(mock_db, mock_http, mock_graph):
    """Voice with no file_size key defaults to 0 → not rejected as too large."""
    with patch("app.channels.telegram._download_file",
               new_callable=AsyncMock, return_value=b"audio"), \
         patch("app.services.stt.transcribe",
               new_callable=AsyncMock, return_value="hola"):
        app = make_app(mock_graph)
        payload = {"message": {
            "chat": {"id": 1}, "from": {"id": 1},
            "voice": {"file_id": "abc"},  # no file_size key
        }}
        r = await _post(app, payload)
    assert r.status_code == 200
    mock_graph.ainvoke.assert_awaited_once()


# ── 7. Network failure edge cases (BUGs) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_send_chat_action_timeout_still_returns_ok(mock_db, mock_graph):
    """sendChatAction timeout is swallowed — handler always returns {"ok": True}."""
    failing = AsyncMock()
    failing.__aenter__ = AsyncMock(return_value=failing)
    failing.__aexit__ = AsyncMock(return_value=None)
    failing.post = AsyncMock(side_effect=httpx.ConnectTimeout("timed out"))

    with patch("app.channels.telegram.httpx.AsyncClient", return_value=failing):
        app = make_app(mock_graph)
        r = await _post(app, text_update())
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_final_send_failure_still_returns_ok(mock_db, mock_graph):
    """_send() network failure is logged but swallowed — Telegram always gets 200."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # sendChatAction — succeeds
            r = MagicMock()
            r.status_code = 200
            return r
        raise httpx.ConnectTimeout("sendMessage failed")

    client_mock = AsyncMock()
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=None)
    client_mock.post = AsyncMock(side_effect=side_effect)

    with patch("app.channels.telegram.httpx.AsyncClient", return_value=client_mock):
        app = make_app(mock_graph)
        r = await _post(app, text_update())
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_voice_over_10mb_no_graph_in_app_state(mock_db, mock_http):
    """Voice > 10 MB returns early before graph access — app.state.graph not required."""
    app = make_app()  # no graph set — proves lazy access path works
    payload = voice_update(file_size=11 * 1024 * 1024)
    r = await _post(app, payload)
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert error_call is not None and "10MB" in str(error_call)


@pytest.mark.asyncio
async def test_graph_not_initialized_sends_service_unavailable(mock_db, mock_http):
    """Normal text message with no graph set → user-facing error, no AttributeError crash."""
    app = make_app()  # no graph set
    r = await _post(app, text_update())
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    send_calls = mock_http.post.call_args_list
    error_call = next((c for c in send_calls if "sendMessage" in str(c)), None)
    assert error_call is not None, "Expected a user-facing error message when graph not initialized"
    assert "Lo siento" in str(error_call) or "no está disponible" in str(error_call)


@pytest.mark.asyncio
async def test_duplicate_update_id_not_processed_twice(mock_db, mock_http, mock_graph):
    """Same update_id sent twice → graph invoked exactly once (dedup)."""
    app = make_app(mock_graph)
    payload = {"update_id": 9999, **text_update()}
    await _post(app, payload)
    await _post(app, payload)
    mock_graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_update_id_different_tenants_both_processed(mock_db, mock_http, mock_graph):
    """update_id is sequential PER BOT, not globally unique. Two different
    tenants sending the same update_id must NOT be deduped against each other —
    regression test for the tenant-scoped dedup key fix."""
    app = make_app(mock_graph)
    payload = {"update_id": 4242, **text_update()}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r1 = await c.post(
            "/webhook/telegram/tenant-a",
            json=payload,
            headers={"x-telegram-bot-api-secret-token": SECRET},
        )
        r2 = await c.post(
            "/webhook/telegram/tenant-b",
            json=payload,
            headers={"x-telegram-bot-api-secret-token": SECRET},
        )

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert mock_graph.ainvoke.await_count == 2


# ── 8. Photo messages ────────────────────────────────────────────────────────────────

# Vision extraction, size gating, uncertainty and failure handling are the
# inbound turn's behaviour, not Telegram's — covered once in tests/test_turn.py.
# What stays here is what the TelegramAdapter itself decides.


@pytest.mark.asyncio
async def test_photo_parsed_as_an_image_ref_at_largest_resolution(mock_db, mock_http, mock_graph):
    """Telegram sends every rendition; the adapter must pick the last (largest)."""
    from app.channels.telegram import TelegramAdapter

    adapter = TelegramAdapter(SLUG, BOT, SECRET)
    body = photo_update()
    body["message"]["photo"] = [
        {"file_id": "small", "file_size": 100},
        {"file_id": "large", "file_size": 4000},
    ]

    [inbound] = await adapter.parse(body)
    assert [(r.id, r.kind, r.size_bytes) for r in inbound.media] == [("large", "image", 4000)]


@pytest.mark.asyncio
async def test_document_parsed_as_a_document_ref(mock_db, mock_http, mock_graph):
    """A PDF order used to fall through to the text branch and be ignored."""
    from app.channels.telegram import TelegramAdapter

    adapter = TelegramAdapter(SLUG, BOT, SECRET)
    body = {"message": {"chat": {"id": 100}, "from": {"id": 42},
                        "document": {"file_id": "d1", "mime_type": "application/pdf"}}}

    [inbound] = await adapter.parse(body)
    assert [(r.id, r.kind) for r in inbound.media] == [("d1", "document")]


# ── 9. Photo albums (media_group_id) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_media_group_photos_combined_into_single_graph_call(mock_db, mock_http, mock_graph):
    """Telegram album (shared media_group_id) arrives as separate updates —
    must be buffered and forwarded as ONE combined graph turn, not N disconnected
    replies."""
    queries = iter(["¿Cuánto cuesta un examen de IGRA?", "¿Cuánto cuesta una biopsia?"])

    async def fake_extract(img_bytes, caption, tenant_slug="", specialization_context=""):
        return next(queries)

    with (
        patch("app.channels.turn.settings.openai_vision_model", "vision-model"),
        patch("app.channels.turn.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.telegram._MEDIA_GROUP_DEBOUNCE", 0.05),
        patch("app.channels.telegram._download_file", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.turn.extract_procedure_query", side_effect=fake_extract),
    ):
        app = make_app(mock_graph)
        r1 = await _post(app, photo_group_update("p1", "grp1"))
        r2 = await _post(app, photo_group_update("p2", "grp1"))
        assert r1.status_code == 200 and r2.status_code == 200
        # Neither individual update triggers the graph — only the flush does.
        mock_graph.ainvoke.assert_not_awaited()
        await asyncio.sleep(0.3)

    mock_graph.ainvoke.assert_awaited_once()
    call_input = mock_graph.ainvoke.call_args[0][0]
    content = call_input["messages"][0].content
    assert "IGRA" in content
    assert "biopsia" in content


@pytest.mark.asyncio
async def test_media_group_multi_sample_query_not_double_bulleted(mock_db, mock_http, mock_graph):
    """Regression: /code-review 2026-07-24 — if one photo in the album is a
    multi-sample document (vision.py returns its own multi-line combined
    question), the media-group join must splice it in as-is, not wrap it in
    an extra "- " bullet that nests its header/sub-bullets inside one bullet."""
    multi_sample_query = (
        "Veo varios ítems distintos en la imagen, ¿cuánto cuesta cada uno?\n"
        "- Epiplón\n- Líquido peritoneal"
    )
    queries = iter(["¿Cuánto cuesta un examen de IGRA?", multi_sample_query])

    async def fake_extract(img_bytes, caption, tenant_slug="", specialization_context=""):
        return next(queries)

    with (
        patch("app.channels.turn.settings.openai_vision_model", "vision-model"),
        patch("app.channels.turn.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.telegram._MEDIA_GROUP_DEBOUNCE", 0.05),
        patch("app.channels.telegram._download_file", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.turn.extract_procedure_query", side_effect=fake_extract),
    ):
        app = make_app(mock_graph)
        await _post(app, photo_group_update("p1", "grp-multi"))
        await _post(app, photo_group_update("p2", "grp-multi"))
        await asyncio.sleep(0.3)

    call_input = mock_graph.ainvoke.call_args[0][0]
    content = call_input["messages"][0].content
    assert "- ¿Cuánto cuesta un examen de IGRA?" in content
    # The multi-sample block's own header must NOT be prefixed with a bullet
    # (it isn't a bare item, it's already a formatted multi-line answer).
    assert "- Veo varios ítems" not in content
    assert "Veo varios ítems distintos en la imagen" in content
    assert "- Epiplón" in content
    assert "- Líquido peritoneal" in content


@pytest.mark.asyncio
async def test_media_group_all_uncertain_sends_single_message(mock_db, mock_http, mock_graph):
    """Every photo in the album is illegible → one combined "can't read" message,
    not one per photo."""
    from app.channels.turn import VISION_UNCERTAIN as _VISION_UNCERTAIN

    with (
        patch("app.channels.turn.settings.openai_vision_model", "vision-model"),
        patch("app.channels.turn.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.telegram._MEDIA_GROUP_DEBOUNCE", 0.05),
        patch("app.channels.telegram._download_file", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.turn.extract_procedure_query", new_callable=AsyncMock,
              return_value=_VISION_UNCERTAIN),
    ):
        app = make_app(mock_graph)
        await _post(app, photo_group_update("p1", "grp2"))
        await _post(app, photo_group_update("p2", "grp2"))
        await asyncio.sleep(0.3)

    mock_graph.ainvoke.assert_not_awaited()
    send_calls = mock_http.post.call_args_list
    send_msgs = [c for c in send_calls if "sendMessage" in str(c)]
    assert len(send_msgs) == 1
    assert "escrib" in str(send_msgs[0]).lower()


@pytest.mark.asyncio
async def test_different_media_groups_not_mixed(mock_db, mock_http, mock_graph):
    """Two distinct albums arriving close together must not merge into one turn."""
    async def fake_extract(img_bytes, caption, tenant_slug="", specialization_context=""):
        return "¿Cuánto cuesta un examen X?"

    with (
        patch("app.channels.turn.settings.openai_vision_model", "vision-model"),
        patch("app.channels.turn.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch("app.channels.telegram._MEDIA_GROUP_DEBOUNCE", 0.05),
        patch("app.channels.telegram._download_file", new_callable=AsyncMock, return_value=b"img"),
        patch("app.channels.turn.extract_procedure_query", side_effect=fake_extract),
    ):
        app = make_app(mock_graph)
        await _post(app, photo_group_update("p1", "grp-a"))
        await _post(app, photo_group_update("p2", "grp-b"))
        await asyncio.sleep(0.3)

    assert mock_graph.ainvoke.await_count == 2
