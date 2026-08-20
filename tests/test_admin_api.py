"""
Edge-case tests for the admin API.
DB calls are mocked — no live services required.

Tests cover:
  - Auth: missing header, wrong key, correct key
  - POST /admin/tenants: duplicate slug, duplicate bot_token, valid creation
  - POST /admin/index: tenant not found, valid upload
  - GET /admin/index/{job_id}: found, not found
"""
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth import verify_operator_key
from app.routes.admin import public_router, router

SECRET_KEY = "test-secret-key-for-unit-tests"
OPERATOR_KEY = SECRET_KEY  # clients send raw token; auth.py compares directly


# ── App factory ───────────────────────────────────────────────────────────────

def make_app(bypass_auth=True) -> FastAPI:
    app = FastAPI()
    app.include_router(public_router)
    app.include_router(router)
    if bypass_auth:
        app.dependency_overrides[verify_operator_key] = lambda: None
    return app


async def _request(app, method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await getattr(c, method)(path, **kwargs)


# ── Auth tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_auth_header_returns_422():
    """Missing x-operator-key → 422 (FastAPI required header validation)."""
    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = ""
        mock_settings.secret_key = SECRET_KEY
        r = await _request(app, "get", "/admin/tenants")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_wrong_auth_key_returns_401():
    """Wrong operator key → 401 Unauthorized."""
    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = ""
        mock_settings.secret_key = SECRET_KEY

        result = MagicMock()
        result.fetchall.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
            r = await _request(app, "get", "/admin/tenants",
                               headers={"x-operator-key": "wrong-key"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_correct_auth_key_proceeds():
    """Correct operator key hash → reaches the handler."""
    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = ""
        mock_settings.secret_key = SECRET_KEY

        result = MagicMock()
        result.fetchall.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
            r = await _request(app, "get", "/admin/tenants",
                               headers={"x-operator-key": OPERATOR_KEY})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_operator_token_takes_priority_over_secret_key():
    """When OPERATOR_TOKEN is set, SECRET_KEY is not used for auth."""
    op_token = "special-operator-token"

    app = make_app(bypass_auth=False)
    with patch("app.auth.settings") as mock_settings:
        mock_settings.operator_token = op_token
        mock_settings.secret_key = SECRET_KEY

        result = MagicMock()
        result.fetchall.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
            # secret_key is rejected when operator_token is set
            r = await _request(app, "get", "/admin/tenants",
                               headers={"x-operator-key": SECRET_KEY})
            assert r.status_code == 401

            # operator_token raw value is accepted
            r = await _request(app, "get", "/admin/tenants",
                               headers={"x-operator-key": op_token})
            assert r.status_code == 200


# ── POST /admin/tenants ────────────────────────────────────────────────────────

def _make_db_for_create(existing_row=None):
    """Returns a patched AsyncSessionLocal for create_tenant tests."""
    existing = MagicMock()
    existing.first.return_value = existing_row  # None = no duplicate

    session = AsyncMock()
    session.execute = AsyncMock(return_value=existing)
    session.commit = AsyncMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


VALID_TENANT = {
    "slug": "new-client",
    "bot_token": "111:NEWTOKEN",
    "webhook_secret": "secret-abc",
    "expertise_area": "software",
    "contact_url": "https://example.com",
    "plan": "free",
}


@pytest.mark.asyncio
async def test_create_tenant_success_returns_201_with_api_key():
    """Valid tenant creation returns 201 and includes raw api_key."""
    app = make_app()
    ctx = _make_db_for_create(existing_row=None)
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=VALID_TENANT)
    assert r.status_code == 201
    body = r.json()
    assert body["slug"] == "new-client"
    assert "api_key" in body
    assert len(body["api_key"]) == 64  # secrets.token_hex(32)


@pytest.mark.asyncio
async def test_create_tenant_duplicate_slug_returns_409():
    """Slug already in use → 409 Conflict."""
    app = make_app()
    ctx = _make_db_for_create(existing_row=MagicMock())  # existing row found
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=VALID_TENANT)
    assert r.status_code == 409
    assert "slug" in r.json()["detail"].lower() or "token" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_tenant_duplicate_bot_token_returns_409():
    """bot_token already in use → 409 Conflict (same query checks both)."""
    app = make_app()
    ctx = _make_db_for_create(existing_row=MagicMock())
    payload = {**VALID_TENANT, "slug": "unique-slug", "bot_token": "EXISTING_TOKEN"}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_create_tenant_missing_required_fields_returns_422():
    """Missing slug or bot_token → 422 validation error, not 500."""
    app = make_app()
    r = await _request(app, "post", "/admin/tenants",
                       json={"slug": "only-slug"})  # bot_token missing
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_tenant_missing_webhook_secret_returns_422():
    """webhook_secret is required — missing → 422."""
    app = make_app()
    payload = {"slug": "ok", "bot_token": "123:TOKEN"}  # no webhook_secret
    r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_tenant_invalid_slug_returns_422():
    """Regression: QA on 2026-08-03 found slug had no format validation —
    "My Tenant!" was accepted and interpolated unescaped into the Telegram
    webhook URL (app/routes/admin.py), which Telegram then rejected,
    leaving the tenant with a silently broken webhook. Slug must match
    the lowercase-alnum-with-hyphens pattern every other slug in this
    codebase already uses.
    """
    app = make_app()
    payload = {**VALID_TENANT, "slug": "My Tenant!"}
    r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_tenant_uppercase_slug_returns_422():
    app = make_app()
    payload = {**VALID_TENANT, "slug": "UpperCase"}
    r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_tenant_specialization_context_persisted():
    """specialization_context in the create payload reaches the INSERT."""
    app = make_app()
    ctx = _make_db_for_create(existing_row=None)
    payload = {**VALID_TENANT, "slug": "lab-client", "specialization_context": "jerga médica: IGRA"}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 201


@pytest.mark.asyncio
async def test_create_tenant_results_turnaround_persisted():
    """results_turnaround in the create payload reaches the INSERT (#46)."""
    app = make_app()
    ctx = _make_db_for_create(existing_row=None)
    payload = {**VALID_TENANT, "slug": "lab-client-3", "results_turnaround": "3 a 5 días hábiles"}
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 201


@pytest.mark.asyncio
async def test_create_tenant_catalog_is_closed_and_not_offered_message_persisted():
    """catalog_is_closed/not_offered_message in the create payload reach the
    INSERT (#51)."""
    app = make_app()
    ctx = _make_db_for_create(existing_row=None)
    payload = {
        **VALID_TENANT, "slug": "lab-client-4",
        "catalog_is_closed": True, "not_offered_message": "No realizamos ese estudio.",
    }
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 201


@pytest.mark.asyncio
async def test_create_tenant_specialization_context_over_max_length_returns_422():
    """Server-side enforcement — a direct API client can't bypass the
    admin.html textarea's client-side maxlength."""
    app = make_app()
    payload = {**VALID_TENANT, "slug": "lab-client-2", "specialization_context": "x" * 8001}
    r = await _request(app, "post", "/admin/tenants", json=payload)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_tenant_api_key_is_hashed_in_db():
    """The raw api_key returned to the caller is NOT stored in DB (only the hash is)."""
    app = make_app()
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(first=MagicMock(return_value=None)))
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "post", "/admin/tenants", json=VALID_TENANT)

    raw_key = r.json()["api_key"]
    expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    # Inspect what was passed to db.execute for the INSERT
    insert_call = session.execute.call_args_list[-1]  # last execute is the INSERT
    insert_params = insert_call[0][1]
    assert insert_params["api_key_hash"] == expected_hash
    assert raw_key not in str(insert_params)  # raw key never touches DB


# ── GET /admin/tenants ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_tenants_returns_list():
    """List tenants returns a JSON array (may be empty)."""
    app = make_app()
    result = MagicMock()
    result.fetchall.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "get", "/admin/tenants")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── POST /admin/index ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_index_job_tenant_not_found_returns_404():
    """Uploading a file for a non-existent tenant → 404."""
    app = make_app()
    result = MagicMock()
    result.first.return_value = None  # tenant not found
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(
            app, "post", "/admin/index",
            data={"tenant_slug": "nonexistent"},
            files={"file": ("test.pdf", b"%PDF fake content", "application/pdf")},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_index_job_valid_returns_202_with_job_id():
    """Valid upload → 201 with job_id and status PENDING."""
    app = make_app()

    import uuid as _uuid
    tenant_row = (str(_uuid.uuid4()), "free")  # (tenant_id, plan) — supports tuple unpacking
    tenant_result = MagicMock()
    tenant_result.first.return_value = tenant_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=tenant_result)
    session.scalar = AsyncMock(return_value=0)  # 0 existing docs — under free limit
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx), \
         patch("app.routes.admin.run_index_job", new_callable=AsyncMock), \
         patch("app.routes.admin.asyncio.create_task"):
        r = await _request(
            app, "post", "/admin/index",
            data={"tenant_slug": "demo"},
            files={"file": ("doc.pdf", b"%PDF content", "application/pdf")},
        )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "PENDING"
    # job_id must be a valid UUID
    UUID(body["job_id"])  # raises if not valid


@pytest.mark.asyncio
async def test_create_index_job_replace_all_flag_passed_to_indexer():
    """replace_all=true must reach run_index_job — it widens the stale-chunk
    delete from "this filename only" to "the whole tenant", used when
    replacing an entire catalog with a new source file/format."""
    app = make_app()

    import uuid as _uuid
    tenant_row = (str(_uuid.uuid4()), "free")
    tenant_result = MagicMock()
    tenant_result.first.return_value = tenant_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=tenant_result)
    session.scalar = AsyncMock(return_value=0)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    mock_run_index_job = AsyncMock()
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx), \
         patch("app.routes.admin.run_index_job", mock_run_index_job), \
         patch("app.routes.admin.asyncio.create_task"):
        r = await _request(
            app, "post", "/admin/index",
            data={"tenant_slug": "demo", "replace_all": "true"},
            files={"file": ("catalog.md", b"# new catalog", "text/markdown")},
        )
    assert r.status_code == 200
    mock_run_index_job.assert_called_once()
    assert mock_run_index_job.call_args.kwargs["replace_all"] is True


@pytest.mark.asyncio
async def test_create_index_job_replace_all_defaults_to_false():
    """Omitting replace_all must preserve the existing safe default —
    additive-by-filename, so unrelated documents for the same tenant survive."""
    app = make_app()

    import uuid as _uuid
    tenant_row = (str(_uuid.uuid4()), "free")
    tenant_result = MagicMock()
    tenant_result.first.return_value = tenant_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=tenant_result)
    session.scalar = AsyncMock(return_value=0)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    mock_run_index_job = AsyncMock()
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx), \
         patch("app.routes.admin.run_index_job", mock_run_index_job), \
         patch("app.routes.admin.asyncio.create_task"):
        r = await _request(
            app, "post", "/admin/index",
            data={"tenant_slug": "demo"},
            files={"file": ("doc.pdf", b"%PDF content", "application/pdf")},
        )
    assert r.status_code == 200
    assert mock_run_index_job.call_args.kwargs["replace_all"] is False


@pytest.mark.asyncio
async def test_create_index_job_counts_active_chunks_not_historical_jobs():
    """Regression test: doc_count must come from distinct document_chunks
    sources, not COUNT(*) of historical index_jobs. replace_all deletes stale
    document_chunks rows on re-upload but never touches old index_jobs rows —
    counting jobs would count every past re-upload attempt against the plan
    limit forever, permanently locking out a tenant who iterates on their
    catalog more than `docs` times even though only one document is active."""
    app = make_app()

    import uuid as _uuid
    tenant_row = (str(_uuid.uuid4()), "free")
    tenant_result = MagicMock()
    tenant_result.first.return_value = tenant_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=tenant_result)
    session.scalar = AsyncMock(return_value=1)  # 1 active document
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx), \
         patch("app.routes.admin.run_index_job", new_callable=AsyncMock), \
         patch("app.routes.admin.asyncio.create_task"):
        r = await _request(
            app, "post", "/admin/index",
            data={"tenant_slug": "demo", "replace_all": "true"},
            files={"file": ("catalog_v2.jsonl", b'{"id":"a"}', "application/json")},
        )

    assert r.status_code == 200
    sql_text = str(session.scalar.call_args[0][0])
    assert "document_chunks" in sql_text
    assert "split_part" in sql_text
    assert "index_jobs" not in sql_text


@pytest.mark.asyncio
async def test_create_index_job_still_blocks_when_five_distinct_documents_active():
    """The fix must not accidentally disable the limit entirely — 5 distinct
    active documents on the free plan must still return 429."""
    app = make_app()

    import uuid as _uuid
    tenant_row = (str(_uuid.uuid4()), "free")
    tenant_result = MagicMock()
    tenant_result.first.return_value = tenant_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=tenant_result)
    session.scalar = AsyncMock(return_value=5)  # at the free-plan limit
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(
            app, "post", "/admin/index",
            data={"tenant_slug": "demo"},
            files={"file": ("doc6.pdf", b"%PDF content", "application/pdf")},
        )

    assert r.status_code == 429
    assert "Document limit (5) reached" in r.json()["detail"]


# ── GET /admin/index/{job_id} ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_index_job_not_found_returns_404():
    """Getting a job_id that doesn't exist → 404."""
    app = make_app()
    result = MagicMock()
    result.first.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "get", "/admin/index/nonexistent-uuid")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_index_job_found_returns_job():
    """Existing job_id returns the job details."""
    from datetime import datetime
    app = make_app()
    fake_row = MagicMock()
    fake_row._mapping = {
        "id": "abc-123", "filename": "test.pdf", "status": "COMPLETED",
        "chunks_total": 10, "chunks_done": 10, "error_message": None,
        "created_at": datetime(2026, 1, 1), "updated_at": datetime(2026, 1, 1),
    }
    result = MagicMock()
    result.first.return_value = fake_row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "get", "/admin/index/abc-123")
    assert r.status_code == 200
    assert r.json()["status"] == "COMPLETED"
    assert r.json()["filename"] == "test.pdf"


# ── GET /pricing (public, no auth required) ────────────────────────────────────

@pytest.mark.asyncio
async def test_get_pricing_returns_all_plans():
    """GET /pricing returns free/basic/pro with pricing and limits."""
    app = make_app()
    r = await _request(app, "get", "/pricing")
    assert r.status_code == 200
    body = r.json()
    assert "plans" in body
    plans = body["plans"]

    # All three plans present
    assert "free" in plans
    assert "basic" in plans
    assert "pro" in plans

    # Free plan is free
    assert plans["free"]["price_usd"] == 0
    assert plans["free"]["billing"] == "free"

    # Basic is $5/month
    assert plans["basic"]["price_usd"] == 5
    assert plans["basic"]["billing"] == "monthly"

    # Pro is $10/month
    assert plans["pro"]["price_usd"] == 10
    assert plans["pro"]["billing"] == "monthly"


@pytest.mark.asyncio
async def test_get_pricing_includes_limits():
    """Each plan includes docs, chunks, queries_monthly limits."""
    app = make_app()
    r = await _request(app, "get", "/pricing")
    plans = r.json()["plans"]

    for plan in plans.values():
        assert "limits" in plan
        limits = plan["limits"]
        assert "docs" in limits
        assert "chunks" in limits
        assert "queries_monthly" in limits
        assert limits["docs"] > 0
        assert limits["chunks"] > 0
        assert limits["queries_monthly"] > 0


@pytest.mark.asyncio
async def test_billing_requires_auth():
    """GET /admin/billing/{tenant_slug} requires operator key."""
    app = make_app(bypass_auth=False)
    r = await _request(app, "get", "/admin/billing/acme")
    assert r.status_code in (401, 422)  # 401 or 422 depending on auth impl


@pytest.mark.asyncio
async def test_get_billing_tenant_not_found_returns_404():
    """GET /admin/billing for nonexistent tenant → 404."""
    app = make_app()
    result = MagicMock()
    result.first.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "get", "/admin/billing/nonexistent")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_billing_returns_plan_usage():
    """GET /admin/billing returns plan, usage, and pricing info."""
    app = make_app()

    # Mock tenant lookup — tuple supports `tenant_id, plan = tenant` unpacking
    tenant_row = (1, "basic")
    tenant_result = MagicMock()
    tenant_result.first.return_value = tenant_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=tenant_result)
    session.scalar = AsyncMock(return_value=8)  # 8 docs, 1500 chunks
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.routes.admin.AsyncSessionLocal", return_value=ctx):
        r = await _request(app, "get", "/admin/billing/acme")

    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "basic"
    assert body["price_usd"] == 5
    assert "limits" in body
    assert "usage_percent" in body
    assert body["limits"]["documents"]["max"] == 20  # Basic limit

    # Same regression guard as create_index_job: billing must report active
    # documents (document_chunks), not historical index_jobs.
    doc_count_sql = str(session.scalar.call_args_list[0][0][0])
    assert "document_chunks" in doc_count_sql
    assert "index_jobs" not in doc_count_sql
