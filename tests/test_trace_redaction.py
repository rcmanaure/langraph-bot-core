"""Unit tests for app.services.trace_redaction.RedactingSpanExporter.

Mocked -- no live Phoenix required. Drives a real FakeListChatModel call
through the real OpenInference LangChain instrumentor (a MagicMock LLM
wouldn't be visible to auto-instrumentation) and inspects the attributes
that actually reach the (in-memory) exporter.
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.services.trace_redaction import RedactingSpanExporter


def _run_llm_call(messages: list[BaseMessage]):
    tracer_provider = TracerProvider()
    inner = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(inner)))
    instrumentor = LangChainInstrumentor()
    # LangChainInstrumentor is a process-wide singleton patch. If another
    # test already left it instrumented (e.g. app.main's module-level
    # create_app() -> _setup_phoenix() -> real LangChainInstrumentor
    # .instrument(), which fires in any env with OBSERVABILITY_ENABLED=true
    # -- true in this dev container, off by default elsewhere), a second
    # .instrument() call here is a silent no-op that keeps patching the
    # OLD tracer_provider, so this test's own `inner` exporter never
    # receives a span. Force a clean slate first; uninstrument() on an
    # already-clean instrumentor is a safe no-op (just logs a warning).
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        FakeListChatModel(responses=["the answer"]).invoke(messages)
    finally:
        instrumentor.uninstrument()
    return inner.get_finished_spans()[0].attributes


def test_system_message_survives_redaction():
    attrs = _run_llm_call([SystemMessage(content="system prompt with retrieved chunks")])
    assert attrs["llm.input_messages.0.message.content"] == "system prompt with retrieved chunks"


def test_user_message_text_is_redacted():
    attrs = _run_llm_call([HumanMessage(content="patient document number 12345678")])
    assert "12345678" not in str(attrs)


def test_user_message_image_is_redacted():
    attrs = _run_llm_call([HumanMessage(content=[
        {"type": "text", "text": "a photo"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
    ])])
    assert "SECRET" not in str(attrs)


def test_assistant_history_message_is_not_redacted():
    attrs = _run_llm_call([AIMessage(content="a past answer")])
    assert attrs["llm.input_messages.0.message.content"] == "a past answer"


def test_model_output_is_not_redacted():
    attrs = _run_llm_call([HumanMessage(content="hi")])
    assert attrs["llm.output_messages.0.message.content"] == "the answer"


def test_raw_input_value_blob_is_dropped():
    """input.value is a serialized dump of the whole message list -- it
    mixes every role's text into one blob, so per-role redaction can't
    clean it (there's no role to key off inside a single string). Dropped
    unconditionally instead."""
    attrs = _run_llm_call([HumanMessage(content="patient document number 12345678")])
    assert attrs["input.value"] == "[redacted]"


def test_chain_span_output_value_is_redacted():
    """A LangGraph node/graph produces its own CHAIN-kind spans distinct
    from the LLM call's own span, and their output.value is the *entire
    updated graph state* re-serialized -- including the requester's
    original message, since state carries the full conversation. Found in
    code review: RedactingSpanExporter originally only touched input.value
    and llm.input_messages, leaving this leak untouched."""
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: list

    def node(state: State) -> dict:
        FakeListChatModel(responses=["the answer"]).invoke(state["messages"])
        return {"messages": state["messages"]}

    tracer_provider = TracerProvider()
    inner = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(inner)))
    instrumentor = LangChainInstrumentor()
    instrumentor.uninstrument()  # see _run_llm_call's comment on why this goes first
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        builder = StateGraph(State)
        builder.add_node("node", node)
        builder.add_edge(START, "node")
        builder.add_edge("node", END)
        builder.compile().invoke(
            {"messages": [HumanMessage(content="patient document number 12345678")]}
        )
    finally:
        instrumentor.uninstrument()

    chain_spans = [
        s for s in inner.get_finished_spans()
        if s.attributes.get("openinference.span.kind") == "CHAIN"
    ]
    assert chain_spans, "expected at least one CHAIN-kind span from the graph/node"
    for s in chain_spans:
        assert "12345678" not in str(s.attributes.get("output.value"))
