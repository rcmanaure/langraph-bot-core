from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

SCHEMA_VERSION = "1.0.0"  # bump when AgentState fields change in a breaking way


class AgentState(TypedDict):
    tenant_id: str        # slug; never the full TenantConfig (secrets not checkpointed)
    thread_id: str        # tenant:{slug}:user:{id}:channel:{channel}(:vN)
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_chunks: list[dict]
    triage_decision: str  # "rag" | "catalog" | "human" | "off_topic"
    answer: str
    blocked: NotRequired[bool]  # set by validate node on injection detection
    is_staff: NotRequired[bool]  # resolved once per turn from the allowlist, never from message text
    # The channel's delivery target for this thread (e.g. Telegram's chat id,
    # which differs from its user id) -- carried so human_control.start() can
    # persist it to conversation_audit the moment a thread escalates, letting
    # an operator reply outside a webhook. See ADR-009 / #37.
    chat_id: NotRequired[str]
    # True only when THIS turn's reply was an unconfirmed-approximation offer
    # awaiting the user's yes/no -- read by the NEXT turn to decide whether a
    # bare rejection escalates. Every generate()/interrupt_node return sets
    # it explicitly (never omits the key) so it cannot survive past the one
    # turn it describes. See ADR-009 / #38.
    awaiting_confirmation: NotRequired[bool]
