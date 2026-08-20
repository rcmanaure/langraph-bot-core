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
    assert result == {"retrieved_chunks": [], "not_offered_verdict": False, "not_offered_max_similarity": None}


@pytest.mark.asyncio
async def test_retrieve_chains_hybrid_search_rerank_and_token_cap():
    state = _state()
    raw_chunks = [{"content": f"chunk {i}"} for i in range(5)]
    reranked = [raw_chunks[3], raw_chunks[1]]
    capped = [raw_chunks[3]]

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
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
    assert result == {"retrieved_chunks": capped, "not_offered_verdict": False, "not_offered_max_similarity": None}


@pytest.mark.asyncio
async def test_retrieve_location_question_inserts_address_chunk_cut_by_rerank():
    """Found live: a compound question ("que examenes hacen y donde estan
    ubicados") reranks as one blended query, cutting the tenant's actual
    address chunk out of the top-k in favor of chunks matching the other
    half of the question. A location-intent message must get that chunk
    back regardless of what the blended rerank decided -- at index 1, not
    index 0: generate.py's _has_confirmed_match() reads chunks[0]'s
    similarity specifically, and this probe's own similarity score has
    nothing to do with how confident the REAL top match is (found live: an
    earlier version put it at index 0 and corrupted that signal, making
    generate() hedge/refuse the whole answer)."""
    state = _state(messages=[HumanMessage(content="que examenes hacen y donde estan ubicados")])
    address_chunk = {"content": "Dirección: Av. 5 de Julio, Edif. Prof. SP."}
    top_match = {"content": "chunk about exams", "similarity": 0.9}
    reranked = [top_match, {"content": "another exam chunk"}]

    async def _fake_retrieve_chunks(db, query, namespace):
        if query == "que examenes hacen y donde estan ubicados":
            return [{"content": "raw pool"}]
        return [address_chunk]  # the location probe query

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(side_effect=_fake_retrieve_chunks)),
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=reranked)),
    ):
        result = await retrieve(state)

    assert result["retrieved_chunks"][0] == top_match
    assert result["retrieved_chunks"][1] == address_chunk
    assert result["retrieved_chunks"][2:] == reranked[1:]


@pytest.mark.asyncio
async def test_retrieve_location_question_does_not_duplicate_an_already_surviving_chunk():
    state = _state(messages=[HumanMessage(content="cual es la direccion")])
    address_chunk = {"content": "Dirección: Av. 5 de Julio, Edif. Prof. SP."}
    reranked = [address_chunk, {"content": "chunk about exams"}]

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[address_chunk])),
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=reranked)),
    ):
        result = await retrieve(state)

    assert result["retrieved_chunks"] == reranked


@pytest.mark.asyncio
async def test_retrieve_non_location_question_never_runs_the_location_probe():
    state = _state(messages=[HumanMessage(content="cuanto cuesta la biopsia")])

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
    ):
        await retrieve(state)

    mock_retrieve.assert_awaited_once()  # only the main query, no location probe


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
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "precio de biopsia de mama"


@pytest.mark.asyncio
async def test_retrieve_uses_previous_question_on_bare_rejection():
    """A bare 'no' answering the bot's approximation offer has no retrievable
    content either — retrieve() must anchor back to the previous question,
    same as a bare confirmation does."""
    state = _state(messages=[
        HumanMessage(content="precio de biopsia de mama"),
        HumanMessage(content="no"),
    ])

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "precio de biopsia de mama"


@pytest.mark.asyncio
async def test_retrieve_does_not_anchor_a_negative_word_inside_a_real_question():
    """A message merely containing a negative word, but carrying a question
    of its own, is unaffected — only a whole-message negative anchors."""
    state = _state(messages=[
        HumanMessage(content="precio de biopsia de mama"),
        HumanMessage(content="no, cuanto cuesta la de pulmon"),
    ])

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "no, cuanto cuesta la de pulmon"


@pytest.mark.asyncio
async def test_retrieve_two_bare_rejections_in_a_row_anchor_to_the_original_question():
    """Regression: found in /code-review. A second bare 'no' answering a
    second approximation must anchor to the ORIGINAL question, not to the
    literal text of the first bare 'no' -- that would reintroduce the exact
    content-free-embedding bug these regexes exist to fix."""
    state = _state(messages=[
        HumanMessage(content="precio de biopsia de mama"),
        HumanMessage(content="no"),
        HumanMessage(content="no"),
    ])

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "precio de biopsia de mama"


@pytest.mark.asyncio
async def test_retrieve_lone_rejection_has_nothing_to_anchor_to():
    """A bare 'no' as the conversation's only human message is left alone —
    there is no previous question to fall back to."""
    state = _state(messages=[HumanMessage(content="no")])

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=_mock_db())),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])) as mock_retrieve,
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.cap_chunks_to_tokens", MagicMock(return_value=[])),
    ):
        await retrieve(state)

    assert mock_retrieve.await_args[0][1] == "no"


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
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
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
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
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
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
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
        patch("app.graph.nodes.retrieve.get_tenant_closed_world_context", AsyncMock(return_value={"expertise_area": "", "catalog_is_closed": False})),
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


# ---------------------------------------------------------------------------
# Closed-world not-offered verdict (#49/ADR-010) — plumbing only, no reply
# change yet (that's #51). Covers: both signals miss, only one misses,
# expansion timeout, expansion exception, expansion succeeds with nothing to
# add.
# ---------------------------------------------------------------------------

def _mock_db_with_lexical_result(found: bool):
    """found=True -> an item-type chunk lexically matches (no miss).
    found=False -> no match (a miss)."""
    result = MagicMock()
    result.first.return_value = object() if found else None
    db = _mock_db()
    db.execute = AsyncMock(return_value=result)
    return db


async def _run_retrieve_closed_world(
    similarities, lexical_found, catalog_is_closed=True,
    specialization="jerga médica", expertise_area="", rewrite_result=("cuanto cuesta la biopsia", True),
):
    state = _state()
    chunks = [{"content": "x", "similarity": s} for s in similarities] if similarities else []
    db = _mock_db_with_lexical_result(lexical_found)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value=specialization)),
        patch(
            "app.graph.nodes.retrieve.get_tenant_closed_world_context",
            AsyncMock(return_value={"expertise_area": expertise_area, "catalog_is_closed": catalog_is_closed}),
        ),
        patch("app.graph.nodes.retrieve._rewrite_query", AsyncMock(return_value=rewrite_result)),
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=chunks)),
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=chunks)),
    ):
        return await retrieve(state)


@pytest.mark.asyncio
async def test_closed_world_verdict_true_when_both_signals_miss():
    result = await _run_retrieve_closed_world(similarities=[0.1], lexical_found=False)
    assert result["not_offered_verdict"] is True


@pytest.mark.asyncio
async def test_closed_world_verdict_false_when_only_similarity_misses():
    """Lexical signal finds a match -> disagreement -> no denial (escalates
    as today, per the two-signal rule)."""
    result = await _run_retrieve_closed_world(similarities=[0.1], lexical_found=True)
    assert result["not_offered_verdict"] is False


@pytest.mark.asyncio
async def test_closed_world_verdict_false_when_only_lexical_misses():
    """Similarity is confirmed (above handoff_threshold) -> disagreement ->
    no denial, even with a lexical miss."""
    result = await _run_retrieve_closed_world(similarities=[0.9], lexical_found=False)
    assert result["not_offered_verdict"] is False


@pytest.mark.asyncio
async def test_closed_world_verdict_false_when_tenant_not_closed():
    """catalog_is_closed=False -> verdict never computed, both signals would
    otherwise agree."""
    result = await _run_retrieve_closed_world(similarities=[0.1], lexical_found=False, catalog_is_closed=False)
    assert result["not_offered_verdict"] is False


@pytest.mark.asyncio
async def test_closed_world_verdict_false_when_expansion_timed_out():
    """_rewrite_query's timeout path returns ran=False -- must block denial
    even though both signals would otherwise agree, since the expanded
    query never actually got a chance to find a synonym match."""
    result = await _run_retrieve_closed_world(
        similarities=[0.1], lexical_found=False, rewrite_result=("cuanto cuesta la biopsia", False),
    )
    assert result["not_offered_verdict"] is False


@pytest.mark.asyncio
async def test_closed_world_verdict_false_when_expansion_raised():
    """Same as the timeout case -- an exception inside _rewrite_query is
    already collapsed to ran=False by that function itself; this confirms
    retrieve() honors that flag rather than re-deciding on its own."""
    result = await _run_retrieve_closed_world(
        similarities=[0.1], lexical_found=False, rewrite_result=("cuanto cuesta la biopsia", False),
    )
    assert result["not_offered_verdict"] is False


@pytest.mark.asyncio
async def test_closed_world_verdict_true_when_expansion_ran_but_added_nothing():
    """Expansion completing with nothing to add (ran=True, query unchanged)
    is NOT the same as never running -- the model looked, per ADR-010, so a
    genuine two-signal miss still denies."""
    result = await _run_retrieve_closed_world(
        similarities=[0.1], lexical_found=False, rewrite_result=("cuanto cuesta la biopsia", True),
    )
    assert result["not_offered_verdict"] is True


@pytest.mark.asyncio
async def test_closed_world_verdict_false_when_not_expansion_grade():
    """Neither specialization_context nor expertise_area set -> expansion
    never even attempted (specialization is falsy, same skip as today) ->
    not expansion-grade -> no denial regardless of what the signals say."""
    result = await _run_retrieve_closed_world(
        similarities=[0.1], lexical_found=False, specialization="", expertise_area="",
    )
    assert result["not_offered_verdict"] is False


@pytest.mark.asyncio
async def test_closed_world_verdict_uses_expertise_area_fallback_for_expansion_grade():
    """specialization_context empty but expertise_area set, tenant is
    catalog_is_closed -> the ADR-010 fallback chain kicks in, expansion runs
    against expertise_area, and it counts as expansion-grade."""
    result = await _run_retrieve_closed_world(
        similarities=[0.1], lexical_found=False, specialization="", expertise_area="laboratorio clínico",
    )
    assert result["not_offered_verdict"] is True


@pytest.mark.asyncio
async def test_non_closed_tenant_expansion_gating_unchanged_when_specialization_empty():
    """A non-closed tenant with empty specialization_context still skips
    expansion entirely, even if expertise_area is set -- the fallback chain
    is scoped to catalog_is_closed tenants only (#49's "no visible behavior
    change" requirement for the other ~100% of tenants)."""
    state = _state()
    db = _mock_db_with_lexical_result(True)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch(
            "app.graph.nodes.retrieve.get_tenant_closed_world_context",
            AsyncMock(return_value={"expertise_area": "laboratorio clínico", "catalog_is_closed": False}),
        ),
        patch("app.graph.nodes.retrieve._rewrite_query", AsyncMock()) as mock_rewrite,
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[])),
    ):
        await retrieve(state)

    mock_rewrite.assert_not_called()
