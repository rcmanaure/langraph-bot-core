"""Unit tests for the content-addressed embedding cache — DB and the
underlying embedder are both mocked."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.services.embedding_cache import CachedEmbeddings, _cache_key

# These tests need settings.embedding_cache_enabled=True to exercise the
# cache read/write path. Skip rather than force it on with a patch — caches
# stay off during live testing on purpose (see docker-compose.dev.yml) to
# keep live runs from picking up stale cached results, and a test that
# quietly overrides the real setting no longer proves anything about how
# this environment actually behaves.
_CACHE_SKIP = pytest.mark.skipif(
    not settings.embedding_cache_enabled, reason="embedding cache disabled in this environment"
)


def _ctx(session):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _row(key, embedding):
    row = MagicMock()
    row.key = key
    row.embedding = json.dumps(embedding)
    return row


def _session(rows):
    result = MagicMock()
    result.fetchall.return_value = rows
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


@_CACHE_SKIP
@pytest.mark.asyncio
async def test_aembed_documents_all_cache_misses_calls_underlying_and_stores():
    underlying = MagicMock()
    underlying.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    session = _session(rows=[])  # nothing cached yet
    with patch("app.services.embedding_cache.AsyncSessionLocal", return_value=_ctx(session)):
        cached = CachedEmbeddings(underlying)
        result = await cached.aembed_documents(["hola", "chau"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    underlying.aembed_documents.assert_awaited_once_with(["hola", "chau"])
    # second execute call is the INSERT ... ON CONFLICT DO NOTHING
    insert_call = session.execute.await_args_list[-1]
    assert "INSERT INTO embedding_cache" in str(insert_call.args[0])
    session.commit.assert_awaited_once()


@_CACHE_SKIP
@pytest.mark.asyncio
async def test_aembed_documents_full_cache_hit_skips_underlying_call():
    underlying = MagicMock()
    underlying.aembed_documents = AsyncMock(return_value=[])

    key = _cache_key("hola")
    session = _session(rows=[_row(key, [0.1, 0.2])])
    with patch("app.services.embedding_cache.AsyncSessionLocal", return_value=_ctx(session)):
        cached = CachedEmbeddings(underlying)
        result = await cached.aembed_documents(["hola"])

    assert result == [[0.1, 0.2]]
    underlying.aembed_documents.assert_not_awaited()


@_CACHE_SKIP
@pytest.mark.asyncio
async def test_aembed_documents_partial_hit_only_embeds_the_miss():
    underlying = MagicMock()
    underlying.aembed_documents = AsyncMock(return_value=[[0.9, 0.9]])

    cached_key = _cache_key("hola")
    session = _session(rows=[_row(cached_key, [0.1, 0.2])])
    with patch("app.services.embedding_cache.AsyncSessionLocal", return_value=_ctx(session)):
        cached = CachedEmbeddings(underlying)
        result = await cached.aembed_documents(["hola", "chau"])

    assert result == [[0.1, 0.2], [0.9, 0.9]]
    underlying.aembed_documents.assert_awaited_once_with(["chau"])


@pytest.mark.asyncio
async def test_aembed_query_delegates_to_aembed_documents():
    underlying = MagicMock()
    underlying.aembed_documents = AsyncMock(return_value=[[0.5, 0.5]])

    session = _session(rows=[])
    with patch("app.services.embedding_cache.AsyncSessionLocal", return_value=_ctx(session)):
        cached = CachedEmbeddings(underlying)
        result = await cached.aembed_query("hola")

    assert result == [0.5, 0.5]


@pytest.mark.asyncio
async def test_aembed_documents_empty_input_short_circuits():
    underlying = MagicMock()
    underlying.aembed_documents = AsyncMock(return_value=[])

    cached = CachedEmbeddings(underlying)
    result = await cached.aembed_documents([])

    assert result == []
    underlying.aembed_documents.assert_not_awaited()


def test_cache_key_differs_by_model_and_content():
    with patch("app.services.embedding_cache.settings") as mock_settings:
        mock_settings.embedding_model = "model-a"
        mock_settings.embedding_dim = 1536
        key_a = _cache_key("hola")

        mock_settings.embedding_model = "model-b"
        key_b = _cache_key("hola")

    assert key_a != key_b
