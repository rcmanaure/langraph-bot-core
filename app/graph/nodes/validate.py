import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.services.security import sanitize_user_input
from app.state import AgentState

logger = logging.getLogger(__name__)

_BLOCKED_MSG = "Mensaje no permitido."


async def validate(state: AgentState) -> dict:
    last = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last is None:
        return {}

    try:
        sanitize_user_input(last.content if isinstance(last.content, str) else "")
    except ValueError:
        # Injection detected — short-circuit graph, never reach LLM
        msg = AIMessage(content=_BLOCKED_MSG)
        # Explicit False, not omitted -- this short-circuits straight to
        # respond, skipping generate() entirely, so nothing else resets a
        # stale True left by last turn's approximation offer (see
        # AgentState.awaiting_confirmation).
        return {"blocked": True, "answer": _BLOCKED_MSG, "messages": [msg], "awaiting_confirmation": False}

    return {}
