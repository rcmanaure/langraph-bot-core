"""Unit tests for the retrieve() node — chains retrieve_chunks() (hybrid
search) -> rerank_chunks() (cross-encoder) -> cap_chunks_to_tokens(). None of
retrieve_chunks, rerank_chunks, or cap_chunks_to_tokens is exercised together
anywhere else — test_rag.py and test_rerank.py test each in isolation with
hand-built inputs, and test_catalog_qa.py/test_nodes.py hand-inject
retrieved_chunks directly into state, bypassing this node entirely."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph.nodes.retrieve import cache_key, retrieve


def _state(tenant_id="tenant-1", messages=None):
    if messages is None:
        messages = [HumanMessage(content="cuanto cuesta la biopsia")]
    return {
        "tenant_id": tenant_id,
        "thread_id": f"tenant:{tenant_id}:user:1:channel:telegram",
        "messages": messages,
        "retrieved_chunks": [],
        "triage_decision": "rag",
        "answer": "",
    }


def _mock_db():
    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)
    return mock_db


@pytest.mark.asyncio
async def test_retrieve_no_human_message_skips_db_entirely():
    """No HumanMessage in state -> _last_human_query returns "" -> must not
    even open a DB session (there's nothing to search for)."""
    state = _state(messages=[])

    with patch("app.graph.nodes.retrieve.AsyncSessionLocal") as mock_session_local:
        result = await retrieve(state)

    mock_session_local.assert_not_called()
    assert result == {"retrieved_chunks": []}


@pytest.mark.asyncio
async def test_retrieve_chains_hybrid_search_rerank_and_token_cap():
    state = _state()
    raw_chunks = [{"content": f"chunk {i}"} for i in range(5)]
    reranked = [raw_chunks[3], raw_chunks[1]]
    capped = [raw_chunks[3]]

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=raw_chunks)) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=reranked)) as mock_rerank,
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=capped)) as mock_cap,
    ):
        result = await retrieve(state)

    mock_retrieve.assert_awaited_once()
    assert mock_retrieve.await_args[0][1] == "cuanto cuesta la biopsia"
    assert mock_retrieve.await_args[0][2] == "tenant-1"

    # rerank_chunks must receive retrieve_chunks' output (not raw state) and
    # the configured top_k_results — not a hardcoded/different value.
    mock_rerank.assert_awaited_once_with(
        "cuanto cuesta la biopsia", raw_chunks, settings.top_k_results
    )

    # cap_chunks_to_tokens must receive rerank_chunks' output (not the raw
    # hybrid-search results) and the configured token budget.
    mock_cap.assert_called_once_with(reranked, settings.retrieval_max_tokens)
    assert result == {"retrieved_chunks": capped}


@pytest.mark.asyncio
async def test_retrieve_uses_previous_question_on_bare_confirmation():
    """A bare 'sí' has no retrievable content of its own — retrieve() must
    search for the PREVIOUS question, not the confirmation text itself."""
    state = _state(messages=[
        HumanMessage(content="precio de biopsia de mama"),
        HumanMessage(content="sí"),
    ])

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "precio de biopsia de mama"


@pytest.mark.asyncio
async def test_retrieve_rewrites_query_when_specialization_set():
    """specialization_context present -> LLM rewrite runs, its output is
    CONCATENATED (not swapped) onto the raw query before hybrid search."""
    from app.schemas.retrieve import RewrittenQuery

    state = _state()
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=RewrittenQuery(query="antro gástrico")
    )

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="Patología")),
        patch("app.graph.nodes.retrieve.get_chat_llm", return_value=mock_llm),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "cuanto cuesta la biopsia antro gástrico"


@pytest.mark.asyncio
async def test_retrieve_rewrite_echoing_raw_query_does_not_duplicate_it():
    """Found live in /qa: when the LLM has nothing useful to add, it can
    echo the raw query back instead of returning an empty string. Naively
    concatenating would send a duplicated query ("X X") to hybrid search."""
    from app.schemas.retrieve import RewrittenQuery

    state = _state()
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=RewrittenQuery(query="cuanto cuesta la biopsia")
    )

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="Patología")),
        patch("app.graph.nodes.retrieve.get_chat_llm", return_value=mock_llm),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "cuanto cuesta la biopsia"


@pytest.mark.asyncio
async def test_retrieve_rewrite_failure_falls_back_to_raw_query():
    """LLM rewrite call fails -> retrieval must still proceed with the raw
    query, never block or raise."""
    state = _state()
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="Patología")),
        patch("app.graph.nodes.retrieve.get_chat_llm", return_value=mock_llm),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "cuanto cuesta la biopsia"


@pytest.mark.asyncio
async def test_retrieve_rewrite_timeout_falls_back_to_raw_query():
    """Regression: found live -- a reasoning-capable OPENAI_MODEL streamed
    tokens continuously for ~9 minutes before failing, and no single read
    gap exceeded the client's own timeout=60, so it never fired. Rewrite
    must bound total call duration itself and fall back, not block the
    user's turn indefinitely."""
    import asyncio

    state = _state()
    mock_llm = MagicMock()

    async def _hangs_forever(*args, **kwargs):
        await asyncio.sleep(10)

    mock_llm.with_structured_output.return_value.ainvoke = _hangs_forever

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="Patología")),
        patch("app.graph.nodes.retrieve.get_chat_llm", return_value=mock_llm),
        patch("app.graph.nodes.retrieve._REWRITE_TIMEOUT_SECONDS", 0.05),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "cuanto cuesta la biopsia"


def test_cache_key_format():
    state = _state(tenant_id="acme")
    assert cache_key(state) == "acme::cuanto cuesta la biopsia"


def test_cache_key_same_question_same_tenant_different_users_collide_by_design():
    """Intentional: cache_key is narrower than the default (whole-state) key
    so two different users asking the same question in the same tenant share
    a cache entry. thread_id/user identity must NOT leak into the key."""
    state_a = _state(tenant_id="acme")
    state_a["thread_id"] = "tenant:acme:user:1:channel:telegram"
    state_b = _state(tenant_id="acme")
    state_b["thread_id"] = "tenant:acme:user:999:channel:whatsapp"

    assert cache_key(state_a) == cache_key(state_b)


def test_cache_key_differs_by_tenant_for_same_question():
    state_a = _state(tenant_id="acme")
    state_b = _state(tenant_id="other-tenant")

    assert cache_key(state_a) != cache_key(state_b)
