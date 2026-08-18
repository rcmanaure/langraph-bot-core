"""
Edge-case tests for the operator API (#37 -- reading a waiting thread and
replying through the channel).
DB calls, the channel adapter, and the graph checkpoint are mocked -- no
live services required. Follows tests/test_admin_api.py's ASGI app factory
and auth-override pattern.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.auth import verify_operator_key
from app.routes.operator import router

SECRET_KEY = "test-secret-key-for-unit-tests"
OPERATOR_KEY = SECRET_KEY  # clients send raw token; auth.py compares directly

THREAD_ID = "tenant:acme:user:42:channel:telegram"
WA_THREAD_ID = "tenant:acme:user:555:channel:whatsapp"


# ── App factory ───────────────────────────────────────────────────────────────

def make_app(bypass_auth=True, graph=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.graph = graph
    if bypass_auth:
        app.dependency_overrides[verify_operator_key] = lambda: None
    return app


async def _request(app, method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await getattr(c, method)(path, **kwargs)


def _ctx(row=None, rows=None):
    """One `async with AsyncSessionLocal() as db:` block's worth of mock."""
    result = MagicMock()
    result.first.return_value = row
    result.fetchall.return_value = rows or []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


# ── Auth ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_thread_rejects_missing_operator_key():
    app = make_app(bypass_auth=False)
    r = await _request(app, "get", f"/operator/threads/{THREAD_ID}")
    assert r.status_code == 422  # FastAPI required-header validation


@pytest.mark.asyncio
async def test_get_thread_rejects_wrong_operator_key():
    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = ""
        mock_settings.secret_key = SECRET_KEY
        r = await _request(app, "get", f"/operator/threads/{THREAD_ID}",
                            headers={"x-operator-key": "wrong-key"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_pending_rejects_wrong_operator_key():
    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = ""
        mock_settings.secret_key = SECRET_KEY
        r = await _request(app, "get", "/operator/pending",
                            headers={"x-operator-key": "wrong-key"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_send_message_rejects_wrong_operator_key():
    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = ""
        mock_settings.secret_key = SECRET_KEY
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            headers={"x-operator-key": "wrong-key"},
                            json={"text": "hola", "author": "María"})
    assert r.status_code == 401


# ── GET /operator/threads/{thread_id} ───────────────────────────────────────

@pytest.mark.asyncio
async def test_get_thread_invalid_thread_id_returns_422():
    app = make_app()
    r = await _request(app, "get", "/operator/threads/not-a-thread-id")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_get_thread_concatenates_bot_conversation_and_human_control_log():
    """The bot's side (LangGraph checkpoint) comes first, then whatever was
    exchanged under human control -- the graph never runs again once a
    thread is escalated, so this order is always correct (see #37)."""
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={
        "messages": [
            HumanMessage(content="cuanto cuesta una biopsia"),
            AIMessage(content="No tengo algo exacto, ¿le sirve una biopsia de mama?"),
        ]
    }))
    app = make_app(graph=graph)

    with patch("app.routes.operator.human_control.thread_messages", new_callable=AsyncMock) as msgs:
        msgs.return_value = [
            {"sender": "user", "content": "no, gracias", "author": None,
             "created_at": datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)},
            {"sender": "operator", "content": "Claro, le explico", "author": "María",
             "created_at": datetime(2026, 8, 18, 12, 1, tzinfo=timezone.utc)},
        ]
        r = await _request(app, "get", f"/operator/threads/{THREAD_ID}")

    assert r.status_code == 200
    body = r.json()
    senders = [m["sender"] for m in body["messages"]]
    assert senders == ["user", "bot", "user", "operator"]
    assert body["messages"][0]["content"] == "cuanto cuesta una biopsia"
    assert body["messages"][3]["author"] == "María"
    assert body["messages"][3]["created_at"] == "2026-08-18T12:01:00+00:00"


@pytest.mark.asyncio
async def test_get_thread_with_no_graph_still_returns_human_control_log():
    app = make_app(graph=None)
    with patch("app.routes.operator.human_control.thread_messages", new_callable=AsyncMock) as msgs:
        msgs.return_value = [{"sender": "user", "content": "hola", "author": None, "created_at": None}]
        r = await _request(app, "get", f"/operator/threads/{THREAD_ID}")

    assert r.status_code == 200
    assert r.json()["messages"] == [{"sender": "user", "content": "hola", "author": None, "created_at": None}]


# ── GET /operator/pending ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pending_reports_tenant_slug_and_wa_window():
    now = datetime.now(timezone.utc)
    rows = [
        MagicMock(_mapping={
            "thread_id": WA_THREAD_ID, "user_id": "555", "channel": "whatsapp",
            "user_message": "hola", "interrupt_started_at": now,
            "tenant_slug": "acme", "last_user_message_at": now - timedelta(hours=1),
        }),
        MagicMock(_mapping={
            "thread_id": THREAD_ID, "user_id": "42", "channel": "telegram",
            "user_message": "hola", "interrupt_started_at": now,
            "tenant_slug": "acme", "last_user_message_at": None,
        }),
    ]
    app = make_app()
    with patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(rows=rows)):
        r = await _request(app, "get", "/operator/pending")

    assert r.status_code == 200
    body = r.json()
    assert body[0]["tenant_slug"] == "acme"
    # ~23h left of the 24h window
    assert 22 * 3600 < body[0]["wa_window_remaining_seconds"] < 23 * 3600
    assert body[1]["wa_window_remaining_seconds"] is None  # non-WhatsApp: no window


# ── POST /operator/threads/{thread_id}/messages ─────────────────────────────

@pytest.mark.asyncio
async def test_send_message_no_open_escalation_returns_404():
    app = make_app()
    with patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=None)):
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_send_message_delivers_through_the_channel_not_the_graph():
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="telegram", user_id="42", chat_id="998877", last_user_message_at=None,
    )
    adapter = AsyncMock()
    adapter.send.return_value = True

    with (
        patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)),
        patch("app.routes.operator.build_adapter", new_callable=AsyncMock, return_value=adapter) as build,
        patch("app.routes.operator.human_control.record_message", new_callable=AsyncMock) as record,
    ):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            json={"text": "Sí, tenemos disponibilidad.", "author": "María"})

    assert r.status_code == 200
    assert r.json()["delivered"] is True
    build.assert_awaited_once_with("acme", "telegram")
    # Delivered through the adapter, to the persisted chat_id -- never the graph.
    sent_inbound, sent_text = adapter.send.await_args.args
    assert sent_inbound.chat_id == "998877"
    assert sent_text == "Sí, tenemos disponibilidad."
    record.assert_awaited_once_with(
        "acme", THREAD_ID, "operator", "Sí, tenemos disponibilidad.", author="María",
    )


@pytest.mark.asyncio
async def test_send_message_reports_delivered_false_without_failing_the_request():
    """An adapter that couldn't deliver (bad credentials, channel API error)
    must not make the operator believe the message reached the user -- but
    it WAS recorded, so this isn't a request failure either (#37)."""
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="telegram", user_id="42", chat_id="998877", last_user_message_at=None,
    )
    adapter = AsyncMock()
    adapter.send.return_value = False

    with (
        patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)),
        patch("app.routes.operator.build_adapter", new_callable=AsyncMock, return_value=adapter),
        patch("app.routes.operator.human_control.record_message", new_callable=AsyncMock) as record,
    ):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})

    assert r.status_code == 200
    assert r.json()["delivered"] is False
    record.assert_awaited_once()  # still recorded -- the operator did act


@pytest.mark.asyncio
async def test_send_message_masks_stored_text_but_delivers_it_unmasked():
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="telegram", user_id="42", chat_id="998877", last_user_message_at=None,
    )
    adapter = AsyncMock()
    adapter.send.return_value = True
    raw_text = "Su cédula 87654321 ya está registrada."

    with (
        patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)),
        patch("app.routes.operator.build_adapter", new_callable=AsyncMock, return_value=adapter),
        patch("app.routes.operator.human_control.record_message", new_callable=AsyncMock) as record,
    ):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            json={"text": raw_text, "author": "María"})

    assert r.status_code == 200
    _, delivered_text = adapter.send.await_args.args
    assert delivered_text == raw_text  # unmasked to the user

    stored_text = record.await_args.args[3]
    assert "87654321" not in stored_text  # masked in storage
    assert "[documento]" in stored_text


@pytest.mark.asyncio
async def test_send_message_falls_back_to_user_id_on_whatsapp_when_chat_id_is_unset():
    """An escalation opened before migration 0015 has chat_id=NULL -- on
    WhatsApp the two are always equal anyway, so this keeps old escalations
    sendable."""
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="whatsapp", user_id="555", chat_id=None,
        last_user_message_at=datetime.now(timezone.utc),
    )
    adapter = AsyncMock()
    adapter.send.return_value = True

    with (
        patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)),
        patch("app.routes.operator.build_adapter", new_callable=AsyncMock, return_value=adapter),
        patch("app.routes.operator.human_control.record_message", new_callable=AsyncMock),
    ):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{WA_THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})

    assert r.status_code == 200
    sent_inbound, _ = adapter.send.await_args.args
    assert sent_inbound.chat_id == "555"


@pytest.mark.asyncio
async def test_send_message_rejects_unknown_chat_id_on_telegram():
    """Unlike WhatsApp, guessing user_id on Telegram could silently misdeliver
    -- the very reason chat_id exists -- so this must be a distinguishable
    rejection, never a silent guess (#37)."""
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="telegram", user_id="42", chat_id=None, last_user_message_at=None,
    )

    with patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})

    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "chat_id_unknown"


@pytest.mark.asyncio
async def test_send_message_outside_wa_window_returns_distinguishable_409():
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="whatsapp", user_id="555", chat_id="555",
        last_user_message_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )

    with patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{WA_THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})

    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "outside_service_window"


@pytest.mark.asyncio
async def test_send_message_no_wa_window_row_returns_409():
    """Never having received a WhatsApp message from this user (no window
    row at all -- last_user_message_at comes back None from the LEFT JOIN)
    must reject the same way an expired window does."""
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="whatsapp", user_id="555", chat_id="555", last_user_message_at=None,
    )

    with patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{WA_THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})

    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "outside_service_window"


@pytest.mark.asyncio
async def test_send_message_within_wa_window_succeeds():
    escalation_row = SimpleNamespace(
        tenant_slug="acme", channel="whatsapp", user_id="555", chat_id="555",
        last_user_message_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    adapter = AsyncMock()
    adapter.send.return_value = True

    with (
        patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)),
        patch("app.routes.operator.build_adapter", new_callable=AsyncMock, return_value=adapter),
        patch("app.routes.operator.human_control.record_message", new_callable=AsyncMock),
    ):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{WA_THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})

    assert r.status_code == 200
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_message_invalid_thread_id_returns_422():
    app = make_app()
    r = await _request(app, "post", "/operator/threads/not-a-thread-id/messages",
                        json={"text": "hola", "author": "María"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_send_message_tenant_not_found_returns_404():
    escalation_row = SimpleNamespace(tenant_slug="acme", channel="telegram", user_id="42", chat_id="998877")
    with (
        patch("app.routes.operator.AsyncSessionLocal", return_value=_ctx(row=escalation_row)),
        patch("app.routes.operator.build_adapter", new_callable=AsyncMock, return_value=None),
    ):
        app = make_app()
        r = await _request(app, "post", f"/operator/threads/{THREAD_ID}/messages",
                            json={"text": "hola", "author": "María"})
    assert r.status_code == 404
