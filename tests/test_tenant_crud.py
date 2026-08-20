"""
Mock-based tests for new tenant CRUD endpoints (T2, T4, T6, T7)
and Telegram webhook helpers (T3).

No live services required — all DB and HTTP calls are mocked.

Coverage:
  PATCH  /admin/tenants/{slug}                — 8 paths
  DELETE /admin/tenants/{slug}                — 6 paths
  POST   /admin/tenants/{slug}/regen-key      — 4 paths
  GET    /admin/tenants/{slug}/webhook-status — 5 paths
  telegram.set_webhook                        — 3 paths
  telegram.delete_webhook                     — 2 paths
  telegram.get_webhook_info                   — 3 paths
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# Stub heavy optional modules that the import chain drags in but tests don't use.
for _mod in (
    "pypdf", "filetype", "tiktoken",
    "langchain_openai", "langgraph", "langgraph.graph",
    "langchain_core.vectorstores", "langchain_core.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# app.services.indexer needs a real async stub for run_index_job
if "app.services.indexer" not in sys.modules:
    _indexer_stub = types.ModuleType("app.services.indexer")
    async def _stub_run_index_job(*args, **kwargs):  # noqa: E301
        pass
    _indexer_stub.run_index_job = _stub_run_index_job
    sys.modules["app.services.indexer"] = _indexer_stub

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.auth import verify_operator_key  # noqa: E402
from app.channels.telegram import delete_webhook, get_webhook_info, set_webhook  # noqa: E402
from app.routes.admin import router  # noqa: E402

# ── App / request helpers ─────────────────────────────────────────────────────

def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_operator_key] = lambda: None
    return app


async def req(app, method, path, **kwargs):
    # Use c.request() so DELETE can carry a json body (AsyncClient.delete() doesn't accept json=)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.request(method.upper(), path, **kwargs)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _tenant(
    id=1, slug="acme", plan="free", expertise_area="consulting",
    specialization_context="", contact_url=None, active=True, bot_token="123:TOKEN",
    webhook_secret="wh-secret", api_key_hash="oldhash",
):
    t = MagicMock()
    t.id = id
    t.slug = slug
    t.plan = plan
    t.expertise_area = expertise_area
    t.specialization_context = specialization_context
    t.contact_url = contact_url
    t.active = active
    t.bot_token = bot_token
    t.webhook_secret = webhook_secret
    t.api_key_hash = api_key_hash
    return t


def _orm_result(row):
    """Mimics db.execute() return value for ORM scalar queries."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = row
    return r


def _raw_result(first_row):
    """Mimics db.execute() return value for raw text queries."""
    r = MagicMock()
    r.first.return_value = first_row
    return r


def _session(**overrides):
    s = AsyncMock()
    s.execute = overrides.get("execute", AsyncMock(return_value=_orm_result(None)))
    s.scalar = overrides.get("scalar", AsyncMock(return_value=0))
    s.commit = AsyncMock()
    s.refresh = AsyncMock()
    s.delete = AsyncMock()
    s.rollback = AsyncMock()
    return s


def _ctx(session):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


# ═════════════════════════════════════════════════════════════════════════════
# PATCH /admin/tenants/{slug}
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_patch_not_found_returns_404():
    """Unknown slug → 404."""
    app = make_app()
    sess = _session(execute=AsyncMock(return_value=_orm_result(None)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "patch", "/admin/tenants/ghost", json={"plan": "pro"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_invalid_plan_returns_422():
    """plan value outside Literal["free","basic","pro"] → 422, no DB hit."""
    app = make_app()
    r = await req(app, "patch", "/admin/tenants/acme", json={"plan": "enterprise"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_success_returns_safe_fields():
    """Successful patch returns plan/active/expertise_area/contact_url/slug."""
    app = make_app()
    t = _tenant(plan="free", active=True)
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=True), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"plan": "basic", "active": True})
    assert r.status_code == 200
    body = r.json()
    assert "slug" in body
    assert "plan" in body
    assert "active" in body


@pytest.mark.asyncio
async def test_patch_specialization_context_round_trips():
    """PATCH with specialization_context persists it and returns it in the
    response — the exact class of bug an earlier draft of this plan's spec
    review caught (field write with no round-trip confirmation)."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"specialization_context": "jerga médica: IGRA, antro gástrico"})
    assert r.status_code == 200
    assert r.json()["specialization_context"] == "jerga médica: IGRA, antro gástrico"
    assert t.specialization_context == "jerga médica: IGRA, antro gástrico"


@pytest.mark.asyncio
async def test_patch_specialization_context_null_clears_to_empty_string():
    """Regression: /code-review 2026-07-24 — admin.html sends JSON null when
    the textarea is cleared (`|| null` payload builder). specialization_context
    is NOT NULL, so writing None raised IntegrityError and returned a
    misleading 409. Null must clear the field to "", not attempt a NULL write."""
    app = make_app()
    t = _tenant(specialization_context="jerga vieja")
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"specialization_context": None})
    assert r.status_code == 200
    assert r.json()["specialization_context"] == ""
    assert t.specialization_context == ""


@pytest.mark.asyncio
async def test_patch_specialization_context_over_max_length_returns_422():
    """Server-side enforcement (Pydantic max_length) — not just the admin.html
    textarea's client-side maxlength, which any direct API client bypasses."""
    app = make_app()
    r = await req(app, "patch", "/admin/tenants/acme",
                  json={"specialization_context": "x" * 8001})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_results_turnaround_round_trips():
    """PATCH with results_turnaround persists it and returns it (#46)."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"results_turnaround": "3 a 5 días hábiles"})
    assert r.status_code == 200
    assert r.json()["results_turnaround"] == "3 a 5 días hábiles"
    assert t.results_turnaround == "3 a 5 días hábiles"


@pytest.mark.asyncio
async def test_patch_results_turnaround_null_clears_to_none():
    """Nullable column (like greeting_message) — clearing must write NULL, so
    generate.py omits the note entirely rather than showing an empty string."""
    app = make_app()
    t = _tenant()
    t.results_turnaround = "vieja nota"
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"results_turnaround": None})
    assert r.status_code == 200
    assert r.json()["results_turnaround"] is None
    assert t.results_turnaround is None


@pytest.mark.asyncio
async def test_patch_catalog_is_closed_and_not_offered_message_round_trip():
    """PATCH with catalog_is_closed/not_offered_message persists and returns
    both (#51)."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"catalog_is_closed": True, "not_offered_message": "No lo hacemos."})
    assert r.status_code == 200
    body = r.json()
    assert body["catalog_is_closed"] is True
    assert body["not_offered_message"] == "No lo hacemos."
    assert t.catalog_is_closed is True
    assert t.not_offered_message == "No lo hacemos."


@pytest.mark.asyncio
async def test_patch_catalog_is_closed_null_coerces_to_false_not_integrity_error():
    """Column is NOT NULL -- an explicit `null` payload (the same shape the
    admin UI's `|| null` clear pattern sends for nullable fields) must
    coerce to False, not attempt a NULL write and surface a misleading 409
    (found in /code-review)."""
    app = make_app()
    t = _tenant()
    t.catalog_is_closed = True
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme", json={"catalog_is_closed": None})
    assert r.status_code == 200
    assert r.json()["catalog_is_closed"] is False
    assert t.catalog_is_closed is False


@pytest.mark.asyncio
async def test_patch_not_offered_message_null_clears_to_none():
    app = make_app()
    t = _tenant()
    t.not_offered_message = "vieja"
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme", json={"not_offered_message": None})
    assert r.status_code == 200
    assert r.json()["not_offered_message"] is None
    assert t.not_offered_message is None


@pytest.mark.asyncio
async def test_patch_response_never_contains_credentials():
    """Response MUST NOT expose bot_token, webhook_secret, or any WA credential."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"plan": "pro", "active": True})
    body = r.json()
    for secret_field in ("bot_token", "webhook_secret", "wa_access_token",
                         "wa_app_secret", "wa_verify_token"):
        assert secret_field not in body, f"{secret_field} must not appear in PATCH response"


@pytest.mark.asyncio
async def test_patch_blank_bot_token_not_sent_to_db():
    """Omitting bot_token from payload → field skipped, no conflict check executed."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=None), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        # No bot_token in payload
        r = await req(app, "patch", "/admin/tenants/acme", json={"plan": "pro", "active": True})
    assert r.status_code == 200
    # execute() called once (tenant load) — conflict check was never run
    assert sess.execute.call_count == 1


@pytest.mark.asyncio
async def test_patch_bot_token_conflict_returns_409():
    """bot_token already used by another tenant → 409."""
    app = make_app()
    t = _tenant(id=1, active=True)
    # First call: load tenant; second call: conflict check finds another tenant
    other_id = MagicMock()
    other_id.scalar_one_or_none.return_value = 99  # another tenant's id
    sess = _session(execute=AsyncMock(side_effect=[_orm_result(t), other_id]))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"bot_token": "456:OTHERTOK", "active": True})
    assert r.status_code == 409
    assert "bot_token" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_patch_deactivate_calls_delete_webhook():
    """Setting active=False on a currently active tenant calls delete_webhook."""
    app = make_app()
    t = _tenant(active=True, bot_token="111:OLD")
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))

    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock) as mock_del, \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock) as mock_set:
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"active": False})

    assert r.status_code == 200
    mock_del.assert_called_once()
    mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_patch_activate_calls_set_webhook():
    """Setting active=True on an inactive tenant triggers set_webhook."""
    app = make_app()
    t = _tenant(active=False)
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))

    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.set_webhook", new_callable=AsyncMock, return_value=True) as mock_set, \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock) as mock_del, \
         patch("app.routes.admin.settings") as mock_cfg:
        mock_cfg.app_domain = "bot.example.com"
        r = await req(app, "patch", "/admin/tenants/acme",
                      json={"active": True})

    assert r.status_code == 200
    mock_set.assert_called_once()
    mock_del.assert_not_called()
    assert r.json()["webhook_registered"] is True


# ═════════════════════════════════════════════════════════════════════════════
# DELETE /admin/tenants/{slug}
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delete_confirm_slug_mismatch_returns_400():
    """confirm_slug ≠ URL slug → 400, no DB hit."""
    app = make_app()
    r = await req(app, "delete", "/admin/tenants/acme",
                  json={"confirm_slug": "typo"})
    assert r.status_code == 400
    assert "confirm_slug" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_not_found_returns_404():
    """Tenant doesn't exist → 404."""
    app = make_app()
    sess = _session(execute=AsyncMock(return_value=_orm_result(None)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "delete", "/admin/tenants/ghost",
                      json={"confirm_slug": "ghost"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_running_jobs_returns_409():
    """Active index jobs prevent deletion → 409."""
    app = make_app()
    t = _tenant()
    sess = _session(
        execute=AsyncMock(return_value=_orm_result(t)),
        scalar=AsyncMock(return_value=2),  # 2 running jobs
    )
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "delete", "/admin/tenants/acme",
                      json={"confirm_slug": "acme"})
    assert r.status_code == 409
    assert "2" in r.json()["detail"]


@pytest.mark.asyncio
async def test_delete_success_returns_200():
    """Valid delete with no running jobs → 200."""
    app = make_app()
    t = _tenant()
    generic = MagicMock()
    sess = _session(
        execute=AsyncMock(side_effect=[
            _orm_result(t),  # tenant load
            generic, generic, generic,  # LangGraph cleanup
        ]),
        scalar=AsyncMock(return_value=0),
    )
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "delete", "/admin/tenants/acme",
                      json={"confirm_slug": "acme"})
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert r.json()["slug"] == "acme"


@pytest.mark.asyncio
async def test_delete_calls_delete_webhook():
    """Successful tenant deletion also calls delete_webhook for cleanup."""
    app = make_app()
    t = _tenant(bot_token="777:BOT")
    generic = MagicMock()
    sess = _session(
        execute=AsyncMock(side_effect=[_orm_result(t), generic, generic, generic]),
        scalar=AsyncMock(return_value=0),
    )
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock) as mock_del:
        await req(app, "delete", "/admin/tenants/acme", json={"confirm_slug": "acme"})
    mock_del.assert_called_once_with("777:BOT")


@pytest.mark.asyncio
async def test_delete_langgraph_cleanup_errors_are_suppressed():
    """Errors during LangGraph row deletion (table missing etc.) are silently swallowed."""
    app = make_app()
    t = _tenant()
    sess = _session(
        execute=AsyncMock(side_effect=[
            _orm_result(t),                     # tenant load
            Exception("table does not exist"),  # checkpoint_writes fails
            MagicMock(),                        # checkpoint_blobs ok
            MagicMock(),                        # checkpoints ok
        ]),
        scalar=AsyncMock(return_value=0),
    )
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.delete_webhook", new_callable=AsyncMock):
        r = await req(app, "delete", "/admin/tenants/acme", json={"confirm_slug": "acme"})
    assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# POST /admin/tenants/{slug}/regen-key
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_regen_not_found_returns_404():
    """Regen for unknown tenant → 404."""
    app = make_app()
    sess = _session(execute=AsyncMock(return_value=_orm_result(None)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/ghost/regen-key")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_regen_returns_new_api_key():
    """Successful regen returns a new api_key in the body."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/regen-key")
    assert r.status_code == 200
    assert "api_key" in r.json()


@pytest.mark.asyncio
async def test_regen_key_is_64_hex_chars():
    """Returned api_key is 64 lowercase hex chars (secrets.token_hex(32))."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/regen-key")
    key = r.json()["api_key"]
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


@pytest.mark.asyncio
async def test_regen_keys_are_unique_across_calls():
    """Two sequential regen calls produce different keys."""
    app = make_app()
    t = _tenant()
    sess = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r1 = await req(app, "post", "/admin/tenants/acme/regen-key")
    sess2 = _session(execute=AsyncMock(return_value=_orm_result(t)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess2)):
        r2 = await req(app, "post", "/admin/tenants/acme/regen-key")
    assert r1.json()["api_key"] != r2.json()["api_key"]


# ═════════════════════════════════════════════════════════════════════════════
# GET /admin/tenants/{slug}/webhook-status
# ═════════════════════════════════════════════════════════════════════════════

def _wh_sess(bot_token):
    """Session mock for webhook-status: raw SQL returns a row with bot_token."""
    row = MagicMock()
    row.bot_token = bot_token
    sess = _session(execute=AsyncMock(return_value=_raw_result(row)))
    return sess


@pytest.mark.asyncio
async def test_webhook_status_not_found_returns_404():
    """Unknown slug → 404."""
    app = make_app()
    sess = _session(execute=AsyncMock(return_value=_raw_result(None)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "get", "/admin/tenants/ghost/webhook-status")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_webhook_status_registered():
    """Telegram returns our exact URL → status 'registered'."""
    app = make_app()
    sess = _wh_sess("123:TOKEN")
    expected_url = "https://bot.example.com/webhook/telegram/acme"
    info = {"ok": True, "result": {"url": expected_url}}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.get_webhook_info", new_callable=AsyncMock, return_value=info), \
         patch("app.routes.admin.settings") as mock_cfg:
        mock_cfg.app_domain = "bot.example.com"
        r = await req(app, "get", "/admin/tenants/acme/webhook-status")
    assert r.status_code == 200
    assert r.json()["status"] == "registered"


@pytest.mark.asyncio
async def test_webhook_status_mismatch_different_url():
    """Telegram has a URL but it points elsewhere → status 'mismatch'."""
    app = make_app()
    sess = _wh_sess("123:TOKEN")
    info = {"ok": True, "result": {"url": "https://old-server.com/webhook/telegram/acme"}}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.get_webhook_info", new_callable=AsyncMock, return_value=info), \
         patch("app.routes.admin.settings") as mock_cfg:
        mock_cfg.app_domain = "new-server.com"
        r = await req(app, "get", "/admin/tenants/acme/webhook-status")
    assert r.status_code == 200
    assert r.json()["status"] == "mismatch"
    assert "old-server.com" in r.json()["url"]


@pytest.mark.asyncio
async def test_webhook_status_unknown_when_no_url():
    """Telegram returns empty URL string → status 'unknown'."""
    app = make_app()
    sess = _wh_sess("123:TOKEN")
    info = {"ok": True, "result": {"url": ""}}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.get_webhook_info", new_callable=AsyncMock, return_value=info), \
         patch("app.routes.admin.settings") as mock_cfg:
        mock_cfg.app_domain = "bot.example.com"
        r = await req(app, "get", "/admin/tenants/acme/webhook-status")
    assert r.status_code == 200
    assert r.json()["status"] == "unknown"


@pytest.mark.asyncio
async def test_webhook_status_error_when_api_fails():
    """Telegram API call fails → status 'error' (200 response, not 5xx)."""
    app = make_app()
    sess = _wh_sess("123:TOKEN")
    info = {"ok": False, "error": "Wrong token"}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.routes.admin.get_webhook_info", new_callable=AsyncMock, return_value=info):
        r = await req(app, "get", "/admin/tenants/acme/webhook-status")
    assert r.status_code == 200
    assert r.json()["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# T3 — telegram helpers: set_webhook, delete_webhook, get_webhook_info
# ═════════════════════════════════════════════════════════════════════════════

def _httpx_ctx(mock_client):
    """Patch httpx.AsyncClient used by telegram helpers."""
    return patch("app.channels.telegram.httpx.AsyncClient",
                 return_value=_async_cm(mock_client))


def _async_cm(obj):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=obj)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


# ── set_webhook ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_webhook_returns_true_on_success():
    """Telegram responds ok:true → returns True."""
    client = AsyncMock()
    client.post = AsyncMock(return_value=MagicMock(
        status_code=200, json=MagicMock(return_value={"ok": True})
    ))
    with _httpx_ctx(client):
        result = await set_webhook("123:TOKEN", "https://example.com/wh", "secret")
    assert result is True


@pytest.mark.asyncio
async def test_set_webhook_returns_false_on_telegram_error():
    """Telegram responds ok:false (e.g. bad URL) → returns False, no exception."""
    client = AsyncMock()
    client.post = AsyncMock(return_value=MagicMock(
        status_code=400, json=MagicMock(return_value={"ok": False, "description": "Bad Request"})
    ))
    with _httpx_ctx(client):
        result = await set_webhook("123:TOKEN", "https://example.com/wh", "secret")
    assert result is False


@pytest.mark.asyncio
async def test_set_webhook_returns_false_on_network_error():
    """Network failure → returns False, does not raise."""
    client = AsyncMock()
    client.post = AsyncMock(side_effect=Exception("connection refused"))
    with _httpx_ctx(client):
        result = await set_webhook("123:TOKEN", "https://example.com/wh", "secret")
    assert result is False


# ── delete_webhook ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_webhook_succeeds_silently():
    """Successful deleteWebhook call completes without returning anything."""
    client = AsyncMock()
    client.post = AsyncMock(return_value=MagicMock(
        status_code=200, json=MagicMock(return_value={"ok": True})
    ))
    with _httpx_ctx(client):
        result = await delete_webhook("123:TOKEN")
    assert result is None  # always None


@pytest.mark.asyncio
async def test_delete_webhook_swallows_network_error():
    """Network failure during deleteWebhook does NOT propagate."""
    client = AsyncMock()
    client.post = AsyncMock(side_effect=Exception("timeout"))
    with _httpx_ctx(client):
        try:
            await delete_webhook("123:TOKEN")
        except Exception:
            pytest.fail("delete_webhook must not raise on network error")


# ── get_webhook_info ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_webhook_info_returns_result_on_success():
    """Successful getWebhookInfo call returns ok:True and result dict."""
    payload = {"ok": True, "result": {"url": "https://example.com/wh", "pending_update_count": 0}}
    client = AsyncMock()
    client.get = AsyncMock(return_value=MagicMock(
        status_code=200, json=MagicMock(return_value=payload)
    ))
    with _httpx_ctx(client):
        info = await get_webhook_info("123:TOKEN")
    assert info["ok"] is True
    assert info["result"]["url"] == "https://example.com/wh"


@pytest.mark.asyncio
async def test_get_webhook_info_returns_ok_false_on_api_error():
    """Telegram returns ok:false (bad token) → ok:False with error string."""
    payload = {"ok": False, "description": "Unauthorized"}
    client = AsyncMock()
    client.get = AsyncMock(return_value=MagicMock(
        status_code=401, json=MagicMock(return_value=payload)
    ))
    with _httpx_ctx(client):
        info = await get_webhook_info("bad:TOKEN")
    assert info["ok"] is False
    assert "error" in info


@pytest.mark.asyncio
async def test_get_webhook_info_returns_ok_false_on_network_error():
    """Network exception → ok:False with error string, no raise."""
    client = AsyncMock()
    client.get = AsyncMock(side_effect=Exception("DNS failure"))
    with _httpx_ctx(client):
        info = await get_webhook_info("123:TOKEN")
    assert info["ok"] is False
    assert "DNS failure" in info["error"]
