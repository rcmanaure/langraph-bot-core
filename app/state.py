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
