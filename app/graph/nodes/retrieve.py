import logging
import re

from langchain_core.messages import HumanMessage

from app.config import settings
from app.db import AsyncSessionLocal
from app.schemas.retrieve import RewrittenQuery
from app.services.llm import get_chat_llm
from app.services.rag import cap_chunks_to_tokens, retrieve_chunks
from app.services.rerank import rerank_chunks
from app.services.tenant_context import get_tenant_specialization
from app.state import AgentState

logger = logging.getLogger(__name__)

# A bare confirmation ("si", "sí", "correcto"...) has no retrievable content of
# its own — it's answering the bot's PREVIOUS question (e.g. an approximation
# offer: "¿Eso es lo que necesitas?"). Embedding it verbatim searches for
# nothing meaningful and returns unrelated chunks, so generate() then can't
# find the item it just offered and contradicts itself. Fall back to the
# previous human query so retrieval stays anchored to what's actually being
# confirmed.
_CONFIRMATION_RE = re.compile(
    r"^\s*(si|sí|s|claro|dale|ok|okay|correcto|exacto|eso mismo|as[ií] es|afirmativo)\s*[.!¡]*\s*$",
    re.IGNORECASE,
)


def _last_human_query(state: AgentState) -> str:
    humans = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    if not humans:
        return ""
    last = humans[-1]
    if len(humans) >= 2 and isinstance(last, str) and _CONFIRMATION_RE.match(last):
        return humans[-2]
    return last


def cache_key(state: AgentState) -> str:
    """Cache key for the retrieve node: same tenant + same question -> same chunks.

    Deliberately narrower than the default (whole-state) key, since state also
    carries thread_id and full message history, which are unique per user and
    would defeat caching for the common case of two different users asking the
    same question.
    """
    return f"{state.get('tenant_id', '')}::{_last_human_query(state)}"


_REWRITE_PROMPT = """\
Expandí la siguiente consulta de un usuario agregando términos o sinónimos formales \
que ayuden a encontrarla en un catálogo de productos/servicios, usando tu propio \
conocimiento del rubro descrito abajo. Devolvé SOLO los términos adicionales \
relevantes (no una oración, no una respuesta, NUNCA repitas la consulta original) \
— se van a concatenar a la consulta original, no a reemplazarla. Si no hay nada \
útil que agregar, devolvé un string vacío.

Rubro del negocio: {specialization}

Consulta del usuario: {query}
"""


async def _rewrite_query(query: str, specialization: str) -> str:
    """LLM-based query expansion using the tenant's free-text specialization
    context — no hardcoded per-vertical glossary/synonym dict. Same
    primary/fallback shape as triage.py's structured-output call: any
    failure falls back to the raw query untouched, never blocks retrieval.
    Result is CONCATENATED to the original query, never replaces it, so a
    bad/hallucinated rewrite only adds noise instead of erasing the user's
    actual words from the retrieval input."""
    llm = get_chat_llm()
    prompt = _REWRITE_PROMPT.format(specialization=specialization, query=query)
    try:
        result: RewrittenQuery = await llm.with_structured_output(RewrittenQuery).ainvoke(
            [HumanMessage(content=prompt)]
        )
        addition = result.query.strip()
        # Defense in depth: the prompt asks for an empty string when there's
        # nothing to add, but if the model echoes the query back anyway,
        # concatenating it would duplicate the original query verbatim
        # instead of adding new terms (found live during /qa — a real
        # response, "cuanto cuesta X cuanto cuesta X").
        if not addition or addition.lower() == query.lower():
            return query
        return f"{query} {addition}"
    except Exception as exc:
        logger.warning("retrieve_rewrite_failed=%s using raw query", exc)
        return query


async def retrieve(state: AgentState) -> dict:
    query = _last_human_query(state)
    if not query:
        return {"retrieved_chunks": []}

    # Vertical-agnostic: skipped entirely (zero extra cost/latency) for
    # tenants without a specialization_context — same no-op-when-empty rule
    # already used by generate.py/vision.py for this field.
    specialization = await get_tenant_specialization(state["tenant_id"])
    if specialization:
        query = await _rewrite_query(query, specialization)

    async with AsyncSessionLocal() as db:
        chunks = await retrieve_chunks(db, query, state["tenant_id"])

    chunks = await rerank_chunks(query, chunks, settings.top_k_results)
    return {"retrieved_chunks": cap_chunks_to_tokens(chunks, settings.retrieval_max_tokens)}
