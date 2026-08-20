from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy

from app.config import settings
from app.graph.nodes.generate import generate
from app.graph.nodes.interrupt import interrupt_node
from app.graph.nodes.prune_history import prune_history
from app.graph.nodes.respond import respond
from app.graph.nodes.retrieve import cache_key as retrieve_cache_key
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.triage import triage
from app.graph.nodes.update_profile import update_profile
from app.graph.nodes.validate import validate
from app.graph.nodes.validate_output import validate_output
from app.state import AgentState

# DB/LLM calls can hit transient errors (rate limits, network blips, connection
# resets) — retry a few times before surfacing to the user.
_RETRIEVE_RETRY = RetryPolicy(max_attempts=3)

# generate.py already falls back to a second model internally before raising,
# and the chat client itself retries transient errors before that. Without a
# tighter cap here, a dual-model outage multiplies into up to 3 full
# primary+fallback cycles with additive backoff on top of the client's own
# retries — capped lower to bound worst-case reply latency for a chat turn.
_GENERATE_RETRY = RetryPolicy(max_attempts=2)


def _route_after_validate(state: AgentState) -> str:
    return "respond" if state.get("blocked") else "triage"


def _route_triage(state: AgentState) -> str:
    # retrieve()+rerank_chunks() are LLM/DB round trips that only pay off when
    # generate() actually reads retrieved_chunks — off_topic/greeting/canned
    # return a fixed reply without looking at chunks, and human hands off
    # before generate() runs at all, so those four skip straight past retrieve.
    d = state.get("triage_decision", "rag")
    if d == "human":
        return "interrupt_node"
    if d in ("off_topic", "greeting", "canned"):
        return "generate"
    return "retrieve"  # "rag" or "catalog"


def build_graph(checkpointer=None, store=None):
    g = StateGraph(AgentState)

    g.add_node("validate", validate)
    g.add_node(
        "retrieve",
        retrieve,
        retry_policy=_RETRIEVE_RETRY,
        cache_policy=CachePolicy(key_func=retrieve_cache_key, ttl=90) if settings.retrieve_cache_enabled else None,
    )
    # triage() swallows every exception internally and always falls back to a
    # default decision, so no retry_policy here would ever fire — see
    # app/graph/nodes/triage.py.
    g.add_node("triage", triage)
    g.add_node("generate", generate, retry_policy=_GENERATE_RETRY)
    g.add_node("validate_output", validate_output)
    g.add_node("interrupt_node", interrupt_node)
    g.add_node("respond", respond)
    g.add_node("update_profile", update_profile)
    g.add_node("prune_history", prune_history)

    g.add_edge(START, "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"triage": "triage", "respond": "respond"},
    )
    g.add_conditional_edges(
        "triage",
        _route_triage,
        {"retrieve": "retrieve", "generate": "generate", "interrupt_node": "interrupt_node"},
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate_output")
    g.add_edge("validate_output", "respond")
    g.add_edge("interrupt_node", "respond")
    # update_profile/prune_history run after every path into respond (including
    # the blocked and human-escalation shortcuts) — the profile should capture
    # escalations too, and messages grow regardless of which branch replied.
    # update_profile must run before prune_history: it reads recent messages
    # to extract the profile update, which prune_history may then discard.
    g.add_edge("respond", "update_profile")
    g.add_edge("update_profile", "prune_history")
    g.add_edge("prune_history", END)

    return g.compile(checkpointer=checkpointer, store=store, cache=InMemoryCache())
