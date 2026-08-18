import logging

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.services import human_control
from app.state import AgentState

logger = logging.getLogger(__name__)


async def interrupt_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id", "")

    # interrupt() re-runs this node from the top on every resume, so opening
    # the escalation must be idempotent — human_control.start() only writes
    # the audit row the first time this thread hits an open interrupt, not
    # on each resume replay.
    await human_control.start(state["tenant_id"], thread_id, state.get("chat_id", ""))

    # Suspend graph — resumes via POST /operator/resume/{thread_id}
    # with Command(resume=operator_text)
    operator_answer: str = interrupt({"type": "needs_human", "thread_id": thread_id})

    return {
        "answer": operator_answer,
        "messages": [AIMessage(content=operator_answer)],
        # Whatever the bot offered before this escalation is moot once a
        # person has answered instead -- explicit False, not omitted, or a
        # stale True from before the handoff could escalate an unrelated
        # rejection turns later (see AgentState.awaiting_confirmation).
        "awaiting_confirmation": False,
    }
