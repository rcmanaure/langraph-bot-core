"""Canned answers: admin CRUD (#47) + the TTL-cached loading service
(app/services/canned.py) — cache-hit/miss and invalidate-on-write coverage,
modeled on tests/test_embedding_cache.py's cache-hit/miss pattern (the
nearest prior art, per the spec's Testing Decisions)."""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

for _mod in (
    "pypdf", "filetype", "tiktoken",
    "langchain_openai", "langgraph", "langgraph.graph",
    "langchain_core.vectorstores", "langchain_core.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

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
from app.models.canned_answer import CannedAnswer  # noqa: E402
from app.routes.admin import router  # noqa: E402
from app.services import canned  # noqa: E402


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_operator_key] = lambda: None
    return app


async def req(app, method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.request(method.upper(), path, **kwargs)


def _session(**overrides):
    s = AsyncMock()
    s.execute = overrides.get("execute", AsyncMock())
    s.scalar = overrides.get("scalar", AsyncMock(return_value=1))
    s.commit = AsyncMock()
    s.refresh = overrides.get("refresh", AsyncMock())
    s.add = MagicMock()  # AsyncSession.add() is synchronous, unlike everything else on the session
    return s


def _ctx(session):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _scalars_result(rows):
    r = MagicMock()
    r.scalars.return_value.all.return_value = rows
    return r


def _scalar_one_result(row):
    r = MagicMock()
    r.scalar_one_or_none.return_value = row
    return r


# ═════════════════════════════════════════════════════════════════════════════
# Admin CRUD
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_unknown_tenant_returns_404():
    app = make_app()
    sess = _session(scalar=AsyncMock(return_value=None))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "get", "/admin/tenants/ghost/canned-answers")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_returns_rows_for_tenant():
    app = make_app()
    row = CannedAnswer(id=1, tenant_id=1, keywords=["horario"], match_mode="any", answer="9-5")
    sess = _session(execute=AsyncMock(return_value=_scalars_result([row])))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "get", "/admin/tenants/acme/canned-answers")
    assert r.status_code == 200
    assert r.json() == [{"id": 1, "keywords": ["horario"], "match_mode": "any", "answer": "9-5"}]


@pytest.mark.asyncio
async def test_create_success_invalidates_cache():
    app = make_app()

    def _refresh(obj):
        obj.id = 1
    sess = _session(refresh=AsyncMock(side_effect=_refresh))

    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.services.canned.invalidate") as mock_invalidate:
        r = await req(app, "post", "/admin/tenants/acme/canned-answers",
                      json={"keywords": ["horario", "hora"], "match_mode": "any", "answer": "Lunes a viernes 9-5"})
    assert r.status_code == 201
    body = r.json()
    assert body["keywords"] == ["horario", "hora"]
    assert body["answer"] == "Lunes a viernes 9-5"
    mock_invalidate.assert_called_once_with("acme")


@pytest.mark.asyncio
async def test_create_empty_keywords_returns_422():
    app = make_app()
    r = await req(app, "post", "/admin/tenants/acme/canned-answers", json={"keywords": [], "answer": "x"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_unknown_tenant_returns_404():
    app = make_app()
    sess = _session(scalar=AsyncMock(return_value=None))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "post", "/admin/tenants/ghost/canned-answers", json={"keywords": ["x"], "answer": "y"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_updates_answer_and_invalidates_cache():
    app = make_app()
    row = CannedAnswer(id=5, tenant_id=1, keywords=["horario"], match_mode="any", answer="old")
    sess = _session(execute=AsyncMock(return_value=_scalar_one_result(row)))

    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.services.canned.invalidate") as mock_invalidate:
        r = await req(app, "patch", "/admin/tenants/acme/canned-answers/5", json={"answer": "new"})
    assert r.status_code == 200
    assert r.json()["answer"] == "new"
    assert row.answer == "new"
    mock_invalidate.assert_called_once_with("acme")


@pytest.mark.asyncio
async def test_patch_not_found_returns_404():
    app = make_app()
    sess = _session(execute=AsyncMock(return_value=_scalar_one_result(None)))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "patch", "/admin/tenants/acme/canned-answers/999", json={"answer": "new"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_success_invalidates_cache():
    app = make_app()
    delete_result = MagicMock()
    delete_result.rowcount = 1
    sess = _session(execute=AsyncMock(return_value=delete_result))

    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)), \
         patch("app.services.canned.invalidate") as mock_invalidate:
        r = await req(app, "delete", "/admin/tenants/acme/canned-answers/5")
    assert r.status_code == 204
    mock_invalidate.assert_called_once_with("acme")


@pytest.mark.asyncio
async def test_delete_not_found_returns_404():
    app = make_app()
    delete_result = MagicMock()
    delete_result.rowcount = 0
    sess = _session(execute=AsyncMock(return_value=delete_result))
    with patch("app.routes.admin.AsyncSessionLocal", return_value=_ctx(sess)):
        r = await req(app, "delete", "/admin/tenants/acme/canned-answers/999")
    assert r.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# app/services/canned.py — TTL cache + invalidation
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _clear_canned_cache():
    canned._cache.clear()
    yield
    canned._cache.clear()


def _fetch_result(rows):
    r = MagicMock()
    r.fetchall.return_value = rows
    return r


def _row(id, keywords, match_mode, answer):
    r = MagicMock()
    r.id, r.keywords, r.match_mode, r.answer = id, keywords, match_mode, answer
    return r


@pytest.mark.asyncio
async def test_get_canned_answers_caches_within_ttl():
    """Second call within the TTL hits the cache, not the DB (verified via
    mock call-count) — same intent as test_embedding_cache.py's
    full-cache-hit-skips-underlying-call test."""
    sess = _session(execute=AsyncMock(return_value=_fetch_result([_row(1, ["horario"], "any", "9-5")])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        first = await canned.get_canned_answers("acme")
        second = await canned.get_canned_answers("acme")

    assert first == second == [{"id": 1, "keywords": ["horario"], "match_mode": "any", "answer": "9-5"}]
    sess.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_canned_answers_query_is_deterministically_ordered():
    """Without ORDER BY, Postgres gives no ordering guarantee, so
    match_canned_answer()'s "first matching row" could flip between turns
    for a tenant with an ambiguous multi-row match (found in /code-review)."""
    sess = _session(execute=AsyncMock(return_value=_fetch_result([])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        await canned.get_canned_answers("acme")

    executed_sql = str(sess.execute.await_args.args[0])
    assert "ORDER BY" in executed_sql.upper()


@pytest.mark.asyncio
async def test_invalidate_forces_refetch_without_waiting_out_ttl():
    """The operator-facing guarantee (#47): editing a row is visible on the
    very next lookup, not after the TTL expires."""
    sess = _session(execute=AsyncMock(return_value=_fetch_result([_row(1, ["horario"], "any", "old")])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        await canned.get_canned_answers("acme")

    canned.invalidate("acme")

    sess2 = _session(execute=AsyncMock(return_value=_fetch_result([_row(1, ["horario"], "any", "new")])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess2)):
        result = await canned.get_canned_answers("acme")

    assert result == [{"id": 1, "keywords": ["horario"], "match_mode": "any", "answer": "new"}]


@pytest.mark.asyncio
async def test_get_canned_answers_refetches_after_ttl_expires():
    sess = _session(execute=AsyncMock(return_value=_fetch_result([_row(1, ["horario"], "any", "9-5")])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        await canned.get_canned_answers("acme")

    # Force the cached entry to look stale without sleeping.
    cached_at, rows = canned._cache["acme"]
    canned._cache["acme"] = (cached_at - canned._TTL_SECONDS - 1, rows)

    sess2 = _session(execute=AsyncMock(return_value=_fetch_result([_row(1, ["horario"], "any", "9-5")])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess2)):
        await canned.get_canned_answers("acme")

    sess2.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_canned_answers_db_failure_degrades_to_empty_list():
    """A DB hiccup must never break the triage turn that calls this on every
    non-shortcut message -- degrades to "no canned answers" instead."""
    with patch("app.services.canned.AsyncSessionLocal", side_effect=ConnectionError("db down")):
        result = await canned.get_canned_answers("acme")
    assert result == []


# ═════════════════════════════════════════════════════════════════════════════
# app/services/canned.py — match_canned_answer (#50)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_match_returns_answer_for_any_mode_single_keyword_hit():
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, ["horario"], "any", "Lunes a viernes 9-5")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", "cual es su horario de atencion")
    assert result == "Lunes a viernes 9-5"


@pytest.mark.asyncio
async def test_match_does_not_fire_on_a_keyword_embedded_in_another_word():
    """Word-boundary match, not raw substring -- "hora" must not fire on
    "ahora" (found in /code-review)."""
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, ["hora"], "any", "9-5")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", "podemos hablar ahora?")
    assert result is None


@pytest.mark.asyncio
async def test_match_fires_when_keyword_starts_and_ends_with_punctuation():
    """A keyword that's a whole pasted Spanish question ("¿...?") starts and
    ends with punctuation, not a word character -- \\b never fires there
    (no \\w/\\W transition at either edge), so this silently never matched
    in production (found live: a canned answer keyed on a full question
    never triggered, the LLM fell through to its own off-topic refusal)."""
    kw = "¿cuántos días puede pasar una muestra en formol antes de llevarla al laboratorio?"
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, [kw], "any", "Respuesta sobre formol")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", kw)
    assert result == "Respuesta sobre formol"


@pytest.mark.asyncio
async def test_match_ignores_accents_and_case():
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, ["ubicación"], "any", "Av. Siempre Viva 742")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", "CUAL ES SU UBICACION")
    assert result == "Av. Siempre Viva 742"


@pytest.mark.asyncio
async def test_match_any_mode_requires_only_one_keyword():
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, ["horario", "hora"], "any", "9-5")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", "a que hora abren")
    assert result == "9-5"


@pytest.mark.asyncio
async def test_match_all_mode_requires_every_keyword():
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, ["tarjeta", "credito"], "all", "Aceptamos tarjeta de crédito")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        no_match = await canned.match_canned_answer("acme", "aceptan tarjeta de debito")
        full_match = await canned.match_canned_answer("acme", "aceptan tarjeta de credito")
    assert no_match is None
    assert full_match == "Aceptamos tarjeta de crédito"


@pytest.mark.asyncio
async def test_match_returns_none_when_no_trigger_matches():
    sess = _session(execute=AsyncMock(
        return_value=_fetch_result([_row(1, ["horario"], "any", "9-5")])
    ))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", "cuanto cuesta la biopsia de pulmon")
    assert result is None


@pytest.mark.asyncio
async def test_match_returns_none_for_tenant_with_no_canned_answers():
    sess = _session(execute=AsyncMock(return_value=_fetch_result([])))
    with patch("app.services.canned.AsyncSessionLocal", return_value=_ctx(sess)):
        result = await canned.match_canned_answer("acme", "cualquier cosa")
    assert result is None
