import logging

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

    # Suspend graph -- resumed by an operator explicitly ending human control
    # (POST /operator/resume/{thread_id}, #39) or by the scheduler
    # auto-expiring an unclaimed escalation (app/scheduler.py). The resume
    # value is discarded: per ADR-009, whatever happened while the thread was
    # held is never folded back into the graph's own message history -- any
    # reply the user should see was already delivered elsewhere (the
    # operator's own messages via #37's send endpoint, or the scheduler's
    # fallback text on auto-expiry), never through this node.
    interrupt({"type": "needs_human", "thread_id": thread_id})

    return {
        # Whatever the bot offered before this escalation is moot once a
        # person has answered instead -- explicit False, not omitted, or a
        # stale True from before the handoff could escalate an unrelated
        # rejection turns later (see AgentState.awaiting_confirmation).
        "awaiting_confirmation": False,
    }
