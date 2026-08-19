import asyncio
import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from app.graph.thread import profile_namespace
from app.schemas.profile import ProfileExtraction
from app.services.llm import get_chat_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

_MAX_TOPICS = 10
_SCHEMA_VERSION = "1.0.0"

# get_chat_llm() shares its model with generate(), called moments earlier in
# the same turn -- back-to-back calls to the same model routinely trip
# OpenRouter's shared (non-BYOK) rate limit on the upstream provider (found
# live, 10/10 reproduced -- see conversation). One short retry is enough:
# this is a same-turn burst against a per-second limit, not sustained
# throttling, so a longer backoff or more attempts would just add latency to
# a call whose failure is already harmless (see the except block below).
_RATE_LIMIT_RETRY_DELAY = 1.5

# Found live: under that same contention, the call doesn't always come back
# as a clean 429 -- sometimes it just never returns (no exception, nothing
# to retry-on-429 or fall into the except below), and the whole turn hung
# behind it until turn.py's own timeout cut it off 45s later. This node
# already promises "best-effort... never let a failure here affect the
# user's reply" -- a bound is what actually keeps that promise instead of
# depending on the client to fail cleanly.
_LLM_CALL_TIMEOUT_SECONDS = 15

_EXTRACT_PROMPT = """\
Given the user's latest message and the assistant's reply, extract:
- new_topic: a short (2-5 word) label for what this exchange was about (e.g. "precio biopsia", "horario atención"). Null if off-topic or unclear.
Reply with the extraction fields only — do not invent information not present in the message.
"""


def _last_human_message(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


async def update_profile(state: AgentState, runtime: Runtime | None = None) -> dict:
    if runtime is None or runtime.store is None:
        return {}

    # Don't feed a flagged (prompt-injection) message into the extraction LLM
    # call — nothing worth extracting from it, and no reason to give injected
    # content another LLM call to try to influence.
    if state.get("blocked"):
        return {}

    query = _last_human_message(state)
    if not query:
        return {}

    try:
        namespace = profile_namespace(state)
        existing = await runtime.store.aget(namespace, "profile")
        profile = dict(existing.value) if existing else {}

        llm = get_chat_llm()
        payload = f"{_EXTRACT_PROMPT}\n\nUser: {query}\nAssistant: {state.get('answer', '')}"
        structured = llm.with_structured_output(ProfileExtraction)
        try:
            extraction: ProfileExtraction = await asyncio.wait_for(
                structured.ainvoke(payload), timeout=_LLM_CALL_TIMEOUT_SECONDS
            )
        except Exception as exc:
            if isinstance(exc, TimeoutError) or getattr(exc, "status_code", None) == 429:
                logger.warning("update_profile_rate_limited retrying_in=%.1fs", _RATE_LIMIT_RETRY_DELAY)
                await asyncio.sleep(_RATE_LIMIT_RETRY_DELAY)
                extraction = await asyncio.wait_for(
                    structured.ainvoke(payload), timeout=_LLM_CALL_TIMEOUT_SECONDS
                )
            else:
                raise

        topics = list(profile.get("topics_of_interest") or [])
        if extraction.new_topic and extraction.new_topic not in topics:
            topics.insert(0, extraction.new_topic)
        profile["topics_of_interest"] = topics[:_MAX_TOPICS]

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
