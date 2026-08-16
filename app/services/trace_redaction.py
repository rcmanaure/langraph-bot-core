import re
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# OpenInference's OPENINFERENCE_HIDE_* env vars are blanket -- hide every
# message's text/images or none, with no per-role knob (confirmed against
# the installed openinference-instrumentation package: TraceConfig takes no
# per-role option). OPENINFERENCE_HIDE_INPUTS is worse than just blanket:
# verified live that setting it suppresses llm.input_messages.* entirely,
# not just the raw input.value blob -- it would erase the system prompt too,
# not only redact the requester's text, so it's never set. That's too coarse
# for this ticket: the system prompt (always llm.input_messages.0, always
# role "system" -- see generate.py's `[SystemMessage(content=system)] +
# trimmed`) carries the retrieved chunks and must stay visible for
# debugging, while the requester's own message text/images must not. This
# exporter wrapper fills that gap:
#   - input.value is dropped unconditionally on every span, LLM or not. It's
#     a raw serialized dump that mixes every role's (or, on a chain-level
#     span, the whole graph state's) text into one blob, so it can't be
#     redacted per-role, and llm.input_messages.* already carries everything
#     it would that isn't the requester's own text.
#   - llm.input_messages.N.message.* is rewritten only for "user"-role
#     messages -- the one place a per-role split is actually possible, and
#     the reason the blanket env vars don't work here.
#   - output.value is dropped on every non-LLM span. Confirmed live: a
#     LangGraph node/graph produces its own CHAIN-kind spans (auto-
#     instrumented, distinct from the LLM call's own span), and their
#     output.value is the *entire updated graph state* serialized back out
#     -- including the requester's original message, since state carries
#     the full conversation. Only an LLM-kind span's output.value is safe to
#     leave alone (that's the model's own answer, deliberately visible).
# Attribute keys verified live against a real instrumented FakeListChatModel
# call and a real instrumented LangGraph graph.
_ROLE_KEY_RE = re.compile(r"^llm\.input_messages\.(\d+)\.message\.role$")
_REDACTED = "[redacted]"


def _redact_span(span: ReadableSpan) -> ReadableSpan:
    attributes = dict(span.attributes or {})
    changed = False

    if "input.value" in attributes:
        attributes["input.value"] = _REDACTED
        changed = True

    if "output.value" in attributes and attributes.get("openinference.span.kind") != "LLM":
        attributes["output.value"] = _REDACTED
        changed = True

    user_indices = {
        m.group(1)
        for key, value in attributes.items()
        if (m := _ROLE_KEY_RE.match(key)) and value == "user"
    }
    for key in list(attributes):
        for idx in user_indices:
            if key.startswith(f"llm.input_messages.{idx}.message.content"):
                attributes[key] = _REDACTED
                changed = True
                break

    if not changed:
        return span

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class RedactingSpanExporter(SpanExporter):
    """Wraps a real SpanExporter, masking requester-authored message text
    and images across every span a turn produces -- LLM call spans and the
    LangGraph node/graph spans wrapping them -- while leaving the system
    prompt (and therefore the retrieved chunks it contains) and the model's
    own answer visible.
    """

    def __init__(self, wrapped: SpanExporter) -> None:
        self._wrapped = wrapped

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._wrapped.export([_redact_span(s) for s in spans])

    def shutdown(self) -> None:
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._wrapped.force_flush(timeout_millis)
