"""Mock-based tests for the staff-member admin endpoints (app/routes/admin.py).

No live services required — all DB calls are mocked.

Coverage:
  GET    /admin/tenants/{slug}/staff
  POST   /admin/tenants/{slug}/staff
  DELETE /admin/tenants/{slug}/staff/{staff_id}
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import IntegrityError

from app.auth import verify_operator_key
from app.routes.admin import router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_operator_key] = lambda: None
    return app


async def req(app, method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.request(method.upper(), path, **kwargs)


def _ctx(session):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _session(**overrides):
    s = AsyncMock()
    s.execute = overrides.get("execute", AsyncMock())
    s.scalar = overrides.get("scalar", AsyncMock(return_value=1))  # tenant_id found by default
    s.add = MagicMock()
    s.commit = overrides.get("commit", AsyncMock())
    s.refresh = AsyncMock()
    s.rollback = AsyncMock()
    return s


# ═════════════════════════════════════════════════════════════════════════════
# GET /admin/tenants/{slug}/staff
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_unknown_tenant_returns_404():
    app = make_app()
    sess = _session(scalar=AsyncMock(return_value=None))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "get", "/admin/tenants/ghost/staff")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_returns_rows_for_tenant():
    app = make_app()
    rows_result = MagicMock()
    rows_result.fetchall.return_value = [
        MagicMock(_mapping={"id": 1, "channel": "telegram", "identifier": "42", "created_at": None}),
    ]
    sess = _session(execute=AsyncMock(return_value=rows_result))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "get", "/admin/tenants/acme/staff")
    assert r.status_code == 200
    assert r.json() == [{"id": 1, "channel": "telegram", "identifier": "42", "created_at": None}]


# ═════════════════════════════════════════════════════════════════════════════
# POST /admin/tenants/{slug}/staff
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_add_unknown_tenant_returns_404():
    app = make_app()
    sess = _session(scalar=AsyncMock(return_value=None))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/ghost/staff",
                      json={"channel": "telegram", "identifier": "42"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_add_valid_staff_member_returns_201():
    app = make_app()
    sess = _session()
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/staff",
                      json={"channel": "telegram", "identifier": "42"})
    assert r.status_code == 201
    body = r.json()
    assert body["channel"] == "telegram"
    assert body["identifier"] == "42"


@pytest.mark.asyncio
async def test_add_invalid_channel_returns_422():
    app = make_app()
    sess = _session()
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/staff",
                      json={"channel": "sms", "identifier": "42"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_add_normalizes_a_pasted_phone_number():
    """A WhatsApp identifier is the raw digits the Cloud API sends (no '+',
    no punctuation) — an operator pasting a formatted phone number must still
    end up stored in that raw form, or resolve_staff can never match it."""
    app = make_app()
    sess = _session()
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/staff",
                      json={"channel": "whatsapp", "identifier": "+1 555-123-4567"})
    assert r.status_code == 201
    assert r.json()["identifier"] == "15551234567"


@pytest.mark.asyncio
async def test_add_identifier_that_normalizes_to_empty_is_rejected():
    """Found in /code-review: min_length=1 on the raw field let "+" through
    (non-empty pre-normalization), but it strips down to "" here — an empty
    stored identifier could match an unparsed inbound user_id defaulting to
    "", granting staff status nobody configured (see ADR-006)."""
    app = make_app()
    sess = _session()
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/staff",
                      json={"channel": "whatsapp", "identifier": "+"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_add_duplicate_returns_409():
    app = make_app()
    sess = _session(commit=AsyncMock(side_effect=IntegrityError("dup", {}, Exception())))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/acme/staff",
                      json={"channel": "telegram", "identifier": "42"})
    assert r.status_code == 409
    sess.rollback.assert_awaited_once()


# ═════════════════════════════════════════════════════════════════════════════
# DELETE /admin/tenants/{slug}/staff/{staff_id}
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_remove_unknown_tenant_returns_404():
    app = make_app()
    sess = _session(scalar=AsyncMock(return_value=None))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "delete", "/admin/tenants/ghost/staff/1")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_remove_unknown_staff_id_returns_404():
    app = make_app()
    delete_result = MagicMock(rowcount=0)
    sess = _session(execute=AsyncMock(return_value=delete_result))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "delete", "/admin/tenants/acme/staff/999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_remove_success_returns_204():
    app = make_app()
    delete_result = MagicMock(rowcount=1)
    sess = _session(execute=AsyncMock(return_value=delete_result))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "delete", "/admin/tenants/acme/staff/1")
    assert r.status_code == 204
    sess.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_scopes_delete_by_tenant_id_not_only_staff_id():
    """A staff row's numeric id alone must not be enough to delete it — the
    DELETE must also filter by this tenant's id, so one tenant's operator
    can't remove another tenant's staff member by guessing an id."""
    app = make_app()
    delete_result = MagicMock(rowcount=1)
    execute_mock = AsyncMock(return_value=delete_result)
    sess = _session(execute=execute_mock, scalar=AsyncMock(return_value=7))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        await req(app, "delete", "/admin/tenants/acme/staff/1")

    params = execute_mock.await_args.args[1]
    assert params == {"id": 1, "tid": 7}
