"""Integration tests for Phoenix observability -- require a live Phoenix
instance (docker compose up phoenix) reachable at localhost:6006/4317.
Mocked, no-network behavior is covered in tests/test_observability.py.

Auto-skips the whole module if Phoenix isn't reachable, rather than hard
failing CI when the optional service isn't up -- consistent with this
codebase's soft-dependency philosophy for Phoenix elsewhere (app/main.py
_setup_phoenix).

Uses langchain_core.language_models.fake_chat_models.FakeListChatModel, not
a bare unittest.mock.MagicMock, deliberately: OpenInference's
LangChainInstrumentor patches methods on real LangChain classes
(BaseChatModel, Runnable). A MagicMock is not an instance of those classes,
so calls through it are invisible to auto-instrumentation -- the existing
_mock_chat_llm() helper in tests/test_graph_integration.py would produce
zero spans if reused here, silently passing a test that verifies nothing.
FakeListChatModel is a real BaseChatModel subclass shipped in langchain_core
specifically for this kind of test: it makes calls the instrumentor can see,
with no network cost.

Tests cover:
  - Redaction: a marker string set as LLM input/output never appears in the
    exported span, once OPENINFERENCE_HIDE_* env vars are set (mirrors
    app.main._verify_redaction's own technique)
  - Happy path / wiring: a real (fake-LLM) LangChain call produces a span in
    Phoenix -- proves auto-instrumentation actually engages, not just that
    register() didn't raise
  - Cache-hit span behavior (D3, CEO plan): does a LangGraph CachePolicy
    cache-hit skip the wrapped node body entirely (and therefore never
    produce a second span), or re-run and produce a second, possibly
    misleadingly-fast span? This was an open question, not a known
    behavior -- the test records the finding rather than asserting a
    pre-decided answer.
"""
import socket
import time
import uuid

import pytest


def _phoenix_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 6006), timeout=1):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _phoenix_reachable(), reason="Phoenix not reachable at localhost:6006"),
]


@pytest.fixture(scope="module")
def phoenix_client():
    import os

    os.environ["OPENINFERENCE_HIDE_INPUT_TEXT"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUT_TEXT"] = "true"
    os.environ["OPENINFERENCE_HIDE_INPUT_IMAGES"] = "true"

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.client import Client
    from phoenix.otel import register

    tracer_provider = register(
        project_name="langraph-bot-v1-test", auto_instrument=False, batch=True
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        yield Client(), tracer_provider
    finally:
        LangChainInstrumentor().uninstrument()


def _query_spans_by_name(client, name: str, retries: int = 5, delay: float = 1.0):
    """Phoenix ingest is async even after force_flush -- poll briefly."""
    for _ in range(retries):
        spans = client.spans.get_spans(
            project_identifier="langraph-bot-v1-test", name=name, limit=10
        )
        if spans:
            return spans
        time.sleep(delay)
    return []


def test_redaction_masks_message_content(phoenix_client):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    client, tracer_provider = phoenix_client
    marker = f"REDACTION_TEST_{uuid.uuid4().hex}"

    llm = FakeListChatModel(responses=[marker])
    llm.invoke(marker)
    tracer_provider.force_flush()

    spans = _query_spans_by_name(client, "FakeListChatModel")
    assert spans, "expected at least one span from the FakeListChatModel call"
    for s in spans:
        # get_spans() returns plain dicts, not span objects -- confirmed by
        # actually running this against live Phoenix.
        assert marker not in str(s["attributes"]), (
            "marker string found unmasked in an exported span -- "
            "OPENINFERENCE_HIDE_* is not working"
        )


def test_spans_appear_for_langchain_calls(phoenix_client):
    """Happy-path / wiring check: proves auto-instrumentation actually
    produces spans, not just that register() didn't raise."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    client, tracer_provider = phoenix_client
    reply = f"wiring-check-{uuid.uuid4().hex}"

    llm = FakeListChatModel(responses=[reply])
    result = llm.invoke("does auto-instrumentation see this call?")
    assert result.content == reply
    tracer_provider.force_flush()

    spans = _query_spans_by_name(client, "FakeListChatModel")
    assert spans, "no span produced for a real LangChain call -- auto-instrumentation is not wired"


def test_cache_hit_span_behavior(phoenix_client):
    """D3 (CEO plan): does a LangGraph CachePolicy cache-hit re-run the
    wrapped node (producing a second, possibly-misleading span) or skip it
    entirely (producing no second span)? Uses a minimal graph, not the full
    bot pipeline (retrieve.py needs a live Postgres this environment
    doesn't have) -- the caching *mechanism* is the same either way, since
    LangGraph's cache sits above the node function, not inside it.

    This records the finding rather than asserting a pre-decided answer --
    the CEO plan explicitly flagged this as unknown, to be verified here.
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.cache.memory import InMemoryCache
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import CachePolicy
    from typing_extensions import TypedDict

    call_count = {"n": 0}

    class State(TypedDict):
        query: str
        reply: str

    def node(state: State) -> dict:
        call_count["n"] += 1
        llm = FakeListChatModel(responses=[f"cache-test-reply-{call_count['n']}"])
        result = llm.invoke(state["query"])
        return {"reply": result.content}

    builder = StateGraph(State)
    builder.add_node("node", node, cache_policy=CachePolicy(ttl=90))
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(cache=InMemoryCache())

    same_input = {"query": "identical query for cache-hit test"}
    graph.invoke(same_input)
    graph.invoke(same_input)  # should be a cache hit if identical input

    client, tracer_provider = phoenix_client
    tracer_provider.force_flush()
    spans = _query_spans_by_name(client, "FakeListChatModel", retries=3)

    # Finding, not an assumption: LangGraph's cache sits above the node
    # function -- on a cache hit, `node()` (and therefore the LLM call
    # inside it) never executes. call_count and span count should agree.
    if call_count["n"] == 1:
        print(
            "\n[D3 finding] Cache-hit SKIPS the node body entirely -- "
            f"node ran once, {len(spans)} span(s) produced. "
            "A cache-hit produces NO span at all, so PR2's p95 analysis "
            "won't see phantom fast/empty spans from cache hits -- but it "
            "also means cache-hit latency (near-zero) is invisible to "
            "Phoenix, not just fast."
        )
    else:
        print(
            "\n[D3 finding] Cache-hit RE-RAN the node body -- "
            f"node ran {call_count['n']} times, {len(spans)} span(s) produced. "
            "PR2's p95 analysis MUST filter cache-hit spans separately, "
            "since they'd misrepresent real LLM latency."
        )

    assert call_count["n"] >= 1
