import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from app.graph.thread import profile_namespace
from app.state import AgentState

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0.0"


def _last_human_message(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


async def update_profile(state: AgentState, runtime: Runtime | None = None) -> dict:
    """Best-effort bookkeeping only — no LLM call. This used to also run a
    structured-output extraction on every turn to populate
    topics_of_interest, but nothing in the codebase ever reads it (checked:
    app/, admin/operator routes and templates, docs/adr, every GitHub issue)
    — a real per-turn LLM call paying for data with no consumer. The key
    stays in the stored profile (existing values carry forward untouched) so
    a future consumer isn't blocked by a schema change; it just stops
    growing until something reads it and justifies paying for extraction
    again. escalated_to_human_count/last_interaction_at cost nothing and stay.
    """
    if runtime is None or runtime.store is None:
        return {}

    if state.get("blocked"):
        return {}

    if not _last_human_message(state):
        return {}

    try:
        namespace = profile_namespace(state)
        existing = await runtime.store.aget(namespace, "profile")
        profile = dict(existing.value) if existing else {}

        profile.setdefault("topics_of_interest", [])
        profile["escalated_to_human_count"] = profile.get("escalated_to_human_count", 0) + (
            1 if state.get("triage_decision") == "human" else 0
        )
        profile["last_interaction_at"] = datetime.now(timezone.utc).isoformat()
        profile["schema_version"] = _SCHEMA_VERSION

        await runtime.store.aput(namespace, "profile", profile)
    except Exception as exc:
        # Best-effort enrichment — never let a failure here affect the user's reply.
        logger.warning("update_profile_failed thread=%s error=%s", state.get("thread_id"), exc)

    return {}
