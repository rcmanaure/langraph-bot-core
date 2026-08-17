"""End-to-end tests for the real compiled graph (build_graph) — every node is
the actual implementation, only true I/O boundaries are mocked (DB session,
embeddings, chat LLM, rerank HTTP call). This is the one place that exercises
retrieve_chunks() -> rerank_chunks() -> generate() together as the graph
actually wires them; every other test either unit-tests one piece in
isolation or mocks the whole graph away at the webhook layer."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.builder import build_graph

TENANT_ROW = MagicMock(expertise_area="diagnóstico histológico", contact_url=None)


def _initial_state(text: str) -> dict:
    return {
        "tenant_id": "acme",
        "thread_id": "tenant:acme:user:1:channel:telegram",
        "messages": [HumanMessage(content=text)],
        "retrieved_chunks": [],
        "triage_decision": "rag",
        "answer": "",
    }


def _db_row(content, source="catalog.jsonl:1", page=1, similarity=0.9):
    row = MagicMock()
    row.content = content
    row.source = source
    row.page = page
    row.similarity = similarity
    return row


def _mock_db_session(fetchall_rows=None, first_row=TENANT_ROW):
    """Single mock DB session reused by both retrieve()'s hybrid-search query
    (needs .fetchall()) and generate()'s tenant lookup (needs .first())."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = fetchall_rows or []
    mock_result.first.return_value = first_row
    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    return db


def _mock_embeddings():
    e = MagicMock()
    e.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    return e


def _mock_chat_llm(reply_text: str, triage_decision: str = "rag"):
    from app.schemas.triage import TriageDecision

    llm = MagicMock()
    llm.model_name = "test-model"
    structured = AsyncMock()
    structured.ainvoke = AsyncMock(return_value=TriageDecision(decision=triage_decision))
    llm.with_structured_output.return_value = structured
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply_text))
    return llm


def _mock_rerank_http_response(results):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"results": results}
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_full_rag_flow_retrieves_reranks_and_answers():
    """The real path: triage classifies 'rag' -> retrieve() runs real
    retrieve_chunks + real rerank_chunks (only httpx mocked) -> generate()
    answers using the reranked chunks -> validate_output -> respond."""
    graph = build_graph(checkpointer=None)
    state = _initial_state("cuanto cuesta la biopsia de pulmon")

    rows = [
        _db_row("SRP009 | Pulmón – PAFF | $90.00", similarity=0.6),
        _db_row("SRP011 | Lobectomía | $240.00", similarity=0.9),
    ]
    # Rerank flips hybrid order: index 1 (Lobectomía) ranked above index 0.
    rerank_client = _mock_rerank_http_response([
        {"index": 1, "relevance_score": 0.95, "document": {"text": ""}},
        {"index": 0, "relevance_score": 0.40, "document": {"text": ""}},
    ])
    llm = _mock_chat_llm("SRP011 Lobectomía cuesta $240.00", triage_decision="rag")
    db = _mock_db_session(fetchall_rows=rows)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.services.rag.get_embeddings", return_value=_mock_embeddings()),
        patch("app.services.rag.settings.rerank_enabled", True),
        # top_k_results must be SMALLER than the candidate count, or
        # rerank_chunks()'s own skip-gate (len(chunks) <= top_k) fires before
        # the API is ever called and hybrid order passes through untouched.
        patch("app.services.rag.settings.top_k_results", 1),
        patch("app.services.rerank.httpx.AsyncClient", return_value=rerank_client),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    # top_k=1 + rerank ranking index 1 (Lobectomía) above index 0 (PAFF) means
    # ONLY Lobectomía should survive into generate()'s prompt. If retrieve()'s
    # wiring to rerank_chunks regressed (wrong arg order, output dropped,
    # hybrid order used instead of reranked order), PAFF would leak in instead.
    system_prompt = llm.ainvoke.call_args[0][0][0].content
    assert "Lobectomía" in system_prompt
    assert "PAFF" not in system_prompt

    assert result["answer"] == "SRP011 Lobectomía cuesta $240.00"
    assert isinstance(result["messages"][-1], AIMessage)


@pytest.mark.asyncio
async def test_retrieval_and_rerank_spans_are_emitted():
    """Neither the raw SQL retrieval query nor the rerank httpx call is a
    LangChain runnable, so OpenInference's auto-instrumentation never
    produces a span for either -- without an explicit one, both show up as
    unattributed dead time between the triage and generate spans. Patches
    each service module's tracer directly (rather than the global OTel
    tracer provider, which real registration -- see app.main.register --
    can only ever be set once per process) so this test doesn't depend on
    whether an earlier test in the same run already claimed it."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    test_tracer = tracer_provider.get_tracer("test")

    graph = build_graph(checkpointer=None)
    state = _initial_state("cuanto cuesta la biopsia de pulmon")
    rows = [
        _db_row("SRP009 | Pulmón – PAFF | $90.00"),
        _db_row("SRP011 | Lobectomía | $240.00"),
    ]
    # top_k_results=1 < len(rows)=2 so rerank_chunks() actually calls the
    # API instead of skipping via its own len(chunks) <= top_k gate.
    rerank_client = _mock_rerank_http_response([
        {"index": 0, "relevance_score": 0.9, "document": {"text": ""}},
        {"index": 1, "relevance_score": 0.4, "document": {"text": ""}},
    ])
    llm = _mock_chat_llm("respuesta", triage_decision="rag")
    db = _mock_db_session(fetchall_rows=rows)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.services.rag.get_embeddings", return_value=_mock_embeddings()),
        patch("app.services.rag._tracer", test_tracer),
        patch("app.services.rag.settings.rerank_enabled", True),
        patch("app.services.rag.settings.top_k_results", 1),
        patch("app.services.rerank.httpx.AsyncClient", return_value=rerank_client),
        patch("app.services.rerank._tracer", test_tracer),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    spans_by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert "chunk_retrieval" in spans_by_name
    assert "rerank" in spans_by_name
    assert spans_by_name["chunk_retrieval"].attributes["openinference.span.kind"] == "RETRIEVER"
    assert spans_by_name["rerank"].attributes["openinference.span.kind"] == "RERANKER"


@pytest.mark.asyncio
async def test_rerank_http_failure_falls_back_but_graph_still_completes():
    """This is the regression-proofing test for the rerank swap: if the
    OpenRouter /rerank call fails mid-graph, rerank_chunks() falls back to
    hybrid order internally — the graph must complete successfully with an
    answer, not propagate the httpx error up through retrieve()'s RetryPolicy
    and fail the whole turn."""
    graph = build_graph(checkpointer=None)
    state = _initial_state("cuanto cuesta la biopsia de pulmon")

    # Two rows + top_k_results=1 below ensures len(chunks) > top_k, so
    # rerank_chunks() actually attempts the API call (and fails) instead of
    # skipping it via its own len(chunks) <= top_k gate.
    rows = [
        _db_row("SRP009 | Pulmón – PAFF | $90.00"),
        _db_row("SRP011 | Lobectomía | $240.00"),
    ]
    failing_client = AsyncMock()
    failing_client.__aenter__ = AsyncMock(return_value=failing_client)
    failing_client.__aexit__ = AsyncMock(return_value=None)
    failing_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    llm = _mock_chat_llm("SRP009 cuesta $90.00", triage_decision="rag")
    db = _mock_db_session(fetchall_rows=rows)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.services.rag.get_embeddings", return_value=_mock_embeddings()),
        patch("app.services.rag.settings.rerank_enabled", True),
        patch("app.services.rag.settings.top_k_results", 1),
        patch("app.services.rerank.httpx.AsyncClient", return_value=failing_client),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    failing_client.post.assert_awaited_once()  # confirms the rerank call was actually attempted, not skipped
    # Fallback keeps hybrid order (PAFF was row 0) truncated to top_k=1 —
    # Lobectomía (row 1) must NOT have survived the fallback slicing.
    system_prompt = llm.ainvoke.call_args[0][0][0].content
    assert "PAFF" in system_prompt
    assert "Lobectomía" not in system_prompt
    assert result["answer"] == "SRP009 cuesta $90.00"


@pytest.mark.asyncio
async def test_off_topic_skips_retrieve_and_rerank_entirely():
    graph = build_graph(checkpointer=None)
    state = _initial_state("quien gano el partido de futbol")
    llm = _mock_chat_llm("", triage_decision="off_topic")
    db = _mock_db_session()

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock()) as mock_retrieve,
        patch("app.services.rerank.httpx.AsyncClient") as mock_rerank_client,
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    mock_retrieve.assert_not_called()
    mock_rerank_client.assert_not_called()
    assert "diagnóstico histológico" in result["answer"]


@pytest.mark.asyncio
async def test_greeting_skips_retrieve_and_second_llm_call():
    graph = build_graph(checkpointer=None)
    state = _initial_state("hola buenas")
    llm = _mock_chat_llm("should not be used as final answer", triage_decision="greeting")
    db = _mock_db_session()

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock()) as mock_retrieve,
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    mock_retrieve.assert_not_called()
    # generate() must use the canned greeting, never calling the chat LLM a
    # second time for a reply (only triage's structured-output call happens).
    llm.ainvoke.assert_not_called()
    assert "Hola" in result["answer"] or "hola" in result["answer"].lower()


@pytest.mark.asyncio
async def test_human_escalation_routes_through_interrupt_skipping_retrieve_and_generate():
    graph = build_graph(checkpointer=None)
    state = _initial_state("quiero hablar con un humano")
    llm = _mock_chat_llm("should not be reached", triage_decision="human")
    interrupt_db = AsyncMock()
    interrupt_result = MagicMock()
    interrupt_result.first.return_value = None  # no open interrupt row yet
    interrupt_db.execute = AsyncMock(return_value=interrupt_result)
    interrupt_db.commit = AsyncMock()
    interrupt_db.__aenter__ = AsyncMock(return_value=interrupt_db)
    interrupt_db.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock()) as mock_retrieve,
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.interrupt.AsyncSessionLocal", MagicMock(return_value=interrupt_db)),
        patch("app.graph.nodes.interrupt.interrupt", MagicMock(return_value="un operador te va a contactar")),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    mock_retrieve.assert_not_called()
    llm.ainvoke.assert_not_called()  # generate() never runs on the human path
    assert result["answer"] == "un operador te va a contactar"


@pytest.mark.asyncio
async def test_human_escalation_actually_suspends_the_graph():
    """The mocked-interrupt() test above proves routing; this one proves
    suspension itself -- interrupt() is real (only the audit DB and
    checkpointer are test doubles), so the graph genuinely pauses instead of
    completing, and ainvoke's result carries "__interrupt__". This is the
    signal app/channels/turn.py reads to send the handoff message instead of
    the empty-answer fallback. See docs/adr/ADR-009-human-control.md."""
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(checkpointer=InMemorySaver())
    state = _initial_state("quiero hablar con un humano")
    llm = _mock_chat_llm("should not be reached", triage_decision="human")
    interrupt_db = AsyncMock()
    interrupt_result = MagicMock()
    interrupt_result.first.return_value = None  # no open interrupt row yet
    interrupt_db.execute = AsyncMock(return_value=interrupt_result)
    interrupt_db.commit = AsyncMock()
    interrupt_db.__aenter__ = AsyncMock(return_value=interrupt_db)
    interrupt_db.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock()) as mock_retrieve,
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.interrupt.AsyncSessionLocal", MagicMock(return_value=interrupt_db)),
    ):
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": state["thread_id"]}}
        )

    mock_retrieve.assert_not_called()
    assert "__interrupt__" in result
    assert not result.get("answer")
