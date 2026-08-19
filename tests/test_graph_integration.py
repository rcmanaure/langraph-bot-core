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
from app.messages import HUMAN_HANDOFF

TENANT_ROW = MagicMock(expertise_area="diagnóstico histológico", contact_url=None, greeting_message=None)


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
    structured.ainvoke = AsyncMock(
        # triage() calls with_structured_output(..., include_raw=True), whose
        # runnable returns the {"raw", "parsed", "parsing_error"} envelope
        # from a single model call.
        return_value={
            "raw": AIMessage(content=""),
            "parsed": TriageDecision(decision=triage_decision),
            "parsing_error": None,
        }
    )
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
    # generate() must use the canned greeting, never calling the chat LLM for
    # a reply. Zero, not one: triage() now gets its raw-fallback text from the
    # same include_raw=True structured call instead of a second concurrent
    # llm.ainvoke(), so nothing on this path touches .ainvoke at all.
    llm.ainvoke.assert_not_called()
    assert "gracias por comunicarte" in result["answer"].lower()


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
        patch("app.services.human_control.AsyncSessionLocal", MagicMock(return_value=interrupt_db)),
        patch("app.graph.nodes.interrupt.interrupt", MagicMock(return_value="un operador te va a contactar")),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    mock_retrieve.assert_not_called()
    # generate() never runs on the human path. Zero, not one: triage() now
    # gets its raw-fallback text from the same include_raw=True structured
    # call instead of a second concurrent llm.ainvoke().
    llm.ainvoke.assert_not_called()
    # The resume value is discarded, never folded into "answer" or the
    # message history (see ADR-009 / #39) -- delivering a reply to the user
    # is #37's job (send through the channel), not this node's.
    assert result["answer"] == ""
    assert not any(m.content == "un operador te va a contactar" for m in result["messages"])


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
        patch("app.services.human_control.AsyncSessionLocal", MagicMock(return_value=interrupt_db)),
    ):
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": state["thread_id"]}}
        )

    mock_retrieve.assert_not_called()
    assert "__interrupt__" in result
    assert not result.get("answer")


@pytest.mark.asyncio
async def test_low_similarity_pool_escalates_through_the_compiled_graph():
    """End to end: retrieve() runs the real hybrid-search query, generate()
    reads its similarity and escalates — the graph completes (no suspend),
    the reply keeps the bot's own answer and closes with the handover
    message, and the escalation opens the same audit row the reactive
    suspend path writes. See docs/adr/ADR-009-human-control.md / #36."""
    graph = build_graph(checkpointer=None)
    state = _initial_state("cuanto cuesta un examen que no existe")

    # A single row well below handoff_threshold (0.30) -- len(chunks)=1 is
    # also <= top_k_results, so rerank_chunks() short-circuits without an
    # HTTP call, keeping this test to the DB/LLM seams only.
    rows = [_db_row("SRP009 | Algo no relacionado | $10.00", similarity=0.1)]
    llm = _mock_chat_llm("No tenemos ese examen en particular.", triage_decision="rag")
    db = _mock_db_session(fetchall_rows=rows)

    audit_db = AsyncMock()
    audit_result = MagicMock()
    audit_result.first.return_value = None  # no open escalation yet
    audit_db.execute = AsyncMock(return_value=audit_result)
    audit_db.commit = AsyncMock()
    audit_db.__aenter__ = AsyncMock(return_value=audit_db)
    audit_db.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.services.rag.get_embeddings", return_value=_mock_embeddings()),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.services.human_control.AsyncSessionLocal", MagicMock(return_value=audit_db)),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    assert "__interrupt__" not in result  # bot answers this turn, graph does not suspend
    assert "No tenemos ese examen en particular." in result["answer"]
    assert HUMAN_HANDOFF in result["answer"]
    audit_db.commit.assert_awaited_once()  # human_control.start() opened the escalation


@pytest.mark.asyncio
async def test_rejection_after_approximation_escalates_across_two_turns():
    """Turn 1: an unconfirmed match, the bot offers an approximation. Turn 2:
    a bare "no" -- retrieval re-anchors to the original query (#34) and
    generate() reads turn 1's awaiting_confirmation off the checkpoint to
    escalate. A real checkpointer, not mocked state, carries that marker
    between the two separate ainvoke() calls. See #38."""
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(checkpointer=InMemorySaver())
    thread_id = "tenant:acme:user:1:channel:telegram"
    config = {"configurable": {"thread_id": thread_id}}

    rows = [_db_row("SRP009 | Algo relacionado | $50.00", similarity=0.5)]
    llm = _mock_chat_llm(
        "Lo más cercano que tenemos es Algo relacionado, ¿es lo que necesita?",
        triage_decision="rag",
    )
    db = _mock_db_session(fetchall_rows=rows)

    audit_db = AsyncMock()
    audit_result = MagicMock()
    audit_result.first.return_value = None  # no open escalation yet
    audit_db.execute = AsyncMock(return_value=audit_result)
    audit_db.commit = AsyncMock()
    audit_db.__aenter__ = AsyncMock(return_value=audit_db)
    audit_db.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.graph.nodes.retrieve.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.graph.nodes.retrieve.get_tenant_specialization", AsyncMock(return_value="")),
        patch("app.services.rag.get_embeddings", return_value=_mock_embeddings()),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
        patch("app.services.human_control.AsyncSessionLocal", MagicMock(return_value=audit_db)),
    ):
        turn1 = await graph.ainvoke(_initial_state("cuanto cuesta algo relacionado"), config=config)
        assert HUMAN_HANDOFF not in turn1["answer"]
        audit_db.commit.assert_not_awaited()

        turn2 = await graph.ainvoke(_initial_state("no"), config=config)

    assert HUMAN_HANDOFF in turn2["answer"]
    audit_db.commit.assert_awaited_once()  # only on turn 2, when the rejection escalates


@pytest.mark.asyncio
async def test_run_turn_interrupt_survives_durability_exit():
    """turn.py calls graph.ainvoke(..., durability="exit") -- checkpoints only
    persist when the graph exits, not after every node. interrupt_node exits
    the graph via a raised GraphInterrupt rather than returning normally, so
    this proves that path still persists under "exit": the real call site
    (run_turn) reaches the user with HUMAN_HANDOFF, and the checkpoint the
    operator's /operator/resume endpoint needs is actually on disk (in this
    test, in the InMemorySaver) afterwards, not silently dropped."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.channels.base import Inbound
    from app.channels.turn import run_turn
    from app.messages import HUMAN_HANDOFF as HANDOFF

    class RecordingAdapter:
        channel = "fake"

        def __init__(self):
            self.sent: list[str] = []

        async def acknowledge(self, inbound):
            pass

        async def send(self, inbound, text):
            self.sent.append(text)
            return True

    inbound = Inbound(
        tenant_slug="acme", channel="fake", user_id="1",
        chat_id="100", message_id="7", text="quiero hablar con un humano",
    )
    thread_id = inbound.thread_id
    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    adapter = RecordingAdapter()
    llm = _mock_chat_llm("should not be reached", triage_decision="human")

    audit_db = AsyncMock()
    audit_result = MagicMock()
    audit_result.first.return_value = None
    audit_db.execute = AsyncMock(return_value=audit_result)
    audit_db.commit = AsyncMock()
    audit_db.__aenter__ = AsyncMock(return_value=audit_db)
    audit_db.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock()) as mock_retrieve,
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.services.human_control.AsyncSessionLocal", MagicMock(return_value=audit_db)),
        patch("app.channels.turn.is_under_human_control", AsyncMock(return_value=False)),
        patch("app.channels.turn.resolve_staff", AsyncMock(return_value=False)),
    ):
        await run_turn(adapter, inbound, graph)

    mock_retrieve.assert_not_called()  # human path never reaches retrieve
    assert adapter.sent == [HANDOFF]

    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    assert state.next == ("interrupt_node",)  # paused there, and it's on disk under "exit"
