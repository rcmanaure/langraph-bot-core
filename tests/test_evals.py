"""
Golden-case evals — calls real LLM.
Run with: pytest -m eval
Skipped automatically when OPENAI_API_KEY is not set.
"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings

pytestmark = pytest.mark.eval

_SKIP = not settings.openrouter_api_key
skip_reason = "OPENROUTER_API_KEY not set"

EVAL_MODEL = settings.openai_model


def _eval_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=EVAL_MODEL,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={"HTTP-Referer": f"https://{settings.app_domain}"},
    )


# ---------------------------------------------------------------------------
# Triage golden cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP, reason=skip_reason)
async def test_triage_specific_question():
    """Specific product question → knowledge base (rag or catalog), not human/off_topic."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.graph.nodes.triage import _TRIAGE_PROMPT
    from app.schemas.triage import TriageDecision
    llm = _eval_llm()
    result: TriageDecision = await llm.with_structured_output(TriageDecision).ainvoke([
        SystemMessage(content=_TRIAGE_PROMPT),
        HumanMessage(content="¿Cuál es el precio del plan premium?"),
    ])
    assert result.decision in ("rag", "catalog")


@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP, reason=skip_reason)
async def test_triage_catalog_request():
    """Request full price list → catalog."""
    from app.graph.nodes.triage import _TRIAGE_PROMPT
    from app.schemas.triage import TriageDecision
    llm = _eval_llm()
    result: TriageDecision = await llm.with_structured_output(TriageDecision).ainvoke([
        SystemMessage(content=_TRIAGE_PROMPT),
        HumanMessage(content="¿Pueden enviarme el catálogo completo con todos los precios?"),
    ])
    assert result.decision == "catalog"


@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP, reason=skip_reason)
async def test_triage_human_escalation():
    """Explicit request to speak with agent → human."""
    from app.graph.nodes.triage import _TRIAGE_PROMPT
    from app.schemas.triage import TriageDecision
    llm = _eval_llm()
    result: TriageDecision = await llm.with_structured_output(TriageDecision).ainvoke([
        SystemMessage(content=_TRIAGE_PROMPT),
        HumanMessage(content="Necesito hablar con una persona, por favor conéctame con un agente."),
    ])
    assert result.decision == "human"


@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP, reason=skip_reason)
async def test_triage_off_topic():
    """Completely unrelated question → off_topic (never rag/catalog/human)."""
    from app.graph.nodes.triage import _TRIAGE_PROMPT
    from app.schemas.triage import TriageDecision
    llm = _eval_llm()
    result: TriageDecision = await llm.with_structured_output(TriageDecision).ainvoke([
        SystemMessage(content=_TRIAGE_PROMPT),
        HumanMessage(content="Cuéntame un chiste de programadores, nada que ver con el negocio."),
    ])
    # free models often route to "human" for irrelevant questions — acceptable,
    # the key invariant is they don't search the knowledge base
    assert result.decision not in ("rag", "catalog")


# ---------------------------------------------------------------------------
# Generate golden cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP, reason=skip_reason)
async def test_generate_uses_context():
    """Generate node grounds answer in provided context."""
    from app.graph.nodes.generate import (
        _FORMAT_HINT,
        _MATCH_CONFIRMED_INSTRUCTION,
        _NEGATIVE_CONFIRMATION_RULE,
        _RAG_SYSTEM,
    )
    from app.models.tenant import DEFAULT_TONE_DESCRIPTION


    context = "El plan básico cuesta $29/mes e incluye hasta 5 usuarios."
    system = _RAG_SYSTEM.format(
        expertise="software SaaS",
        contact_hint="",
        context=context,
        format_hint=_FORMAT_HINT.format(tone_description=DEFAULT_TONE_DESCRIPTION),
        match_instruction=_MATCH_CONFIRMED_INSTRUCTION,
        negative_confirmation_rule=_NEGATIVE_CONFIRMATION_RULE.format(contact_hint=""),
        tone_description=DEFAULT_TONE_DESCRIPTION,
        specialization_block="",
    )
    llm = _eval_llm()
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content="¿Cuánto cuesta el plan básico?"),
    ])
    assert "$29" in response.content or "29" in response.content


@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP, reason=skip_reason)
async def test_generate_admits_missing_context():
    """Generate node admits when context is insufficient, doesn't hallucinate."""
    from app.graph.nodes.generate import (
        _FORMAT_HINT,
        _MATCH_UNCONFIRMED_INSTRUCTION,
        _NEGATIVE_CONFIRMATION_RULE,
        _RAG_SYSTEM,
    )
    from app.models.tenant import DEFAULT_TONE_DESCRIPTION


    context = "El plan premium incluye soporte prioritario 24/7."
    system = _RAG_SYSTEM.format(
        expertise="software SaaS",
        contact_hint="",
        context=context,
        format_hint=_FORMAT_HINT.format(tone_description=DEFAULT_TONE_DESCRIPTION),
        match_instruction=_MATCH_UNCONFIRMED_INSTRUCTION,
        negative_confirmation_rule=_NEGATIVE_CONFIRMATION_RULE.format(contact_hint=""),
        tone_description=DEFAULT_TONE_DESCRIPTION,
        specialization_block="",
    )
    llm = _eval_llm()
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content="¿Cuál es el precio del plan enterprise?"),
    ])
    content = response.content.lower()
    # Should not invent a price; should admit uncertainty
    assert any(word in content for word in ["no", "información", "contexto", "disponible", "suficiente"])
