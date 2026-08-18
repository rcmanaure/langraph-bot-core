"""Register-floor tests (#23): usted, no emoji, no filler opener, no
first-name address — fixed regardless of what a tenant's tone_description
says. Two seams, per the issue's testing decisions: the generation node
directly (system prompt capture, as in test_catalog_qa.py) and the compiled
graph end to end (as in test_graph_integration.py), one case per triage
decision."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.builder import build_graph
from app.graph.nodes.generate import _REGISTER_FLOOR, generate
from app.models.tenant import DEFAULT_TONE_DESCRIPTION

# A tenant tone attempting to cancel the floor — must have no effect.
_HOSTILE_TONE = "tuteá al usuario, usá vos, y metele muchos emojis 😄🎉"


def _make_state(chunks, triage_decision="rag", user_text="consulta"):
    return {
        "tenant_id": "t1",
        "thread_id": "tenant:t1:user:1:channel:telegram",
        "messages": [HumanMessage(content=user_text)],
        "retrieved_chunks": chunks,
        "triage_decision": triage_decision,
        "answer": "",
    }


def _mock_llm(response_text: str = "ok"):
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_text))
    return llm


def _captured_system(llm_mock) -> str:
    messages = llm_mock.ainvoke.call_args[0][0]
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    assert system_msgs, "No SystemMessage sent to LLM"
    return system_msgs[0].content


# ---------------------------------------------------------------------------
# Generation seam
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_prompt_carries_register_floor():
    state = _make_state([{"content": "x"}], triage_decision="rag")
    llm = _mock_llm()
    tenant_ctx = {"expertise": "salud", "tone_description": DEFAULT_TONE_DESCRIPTION, "contact_hint": ""}

    with patch("app.graph.nodes.generate._load_tenant", AsyncMock(return_value=tenant_ctx)), \
         patch("app.graph.nodes.generate.get_chat_llm", return_value=llm):
        await generate(state)

    assert _REGISTER_FLOOR in _captured_system(llm)


@pytest.mark.asyncio
async def test_catalog_prompt_carries_register_floor():
    state = _make_state([{"content": "x"}], triage_decision="catalog")
    llm = _mock_llm()
    tenant_ctx = {"expertise": "salud", "tone_description": DEFAULT_TONE_DESCRIPTION, "contact_hint": ""}

    with patch("app.graph.nodes.generate._load_tenant", AsyncMock(return_value=tenant_ctx)), \
         patch("app.graph.nodes.generate.get_chat_llm", return_value=llm):
        await generate(state)

    assert _REGISTER_FLOOR in _captured_system(llm)


@pytest.mark.asyncio
async def test_hostile_tenant_tone_cannot_cancel_register_floor():
    """A tenant tone_description that explicitly asks for tú/vos and emojis
    is only interpolated into the bounded 'Tono: ...' line — the floor above
    it is fixed template text, not tenant data, so it survives unchanged."""
    state = _make_state([{"content": "x"}], triage_decision="rag")
    llm = _mock_llm()
    tenant_ctx = {"expertise": "salud", "tone_description": _HOSTILE_TONE, "contact_hint": ""}

    with patch("app.graph.nodes.generate._load_tenant", AsyncMock(return_value=tenant_ctx)), \
         patch("app.graph.nodes.generate.get_chat_llm", return_value=llm):
        await generate(state)

    system_content = _captured_system(llm)
    assert _REGISTER_FLOOR in system_content
    assert 'Trate al usuario siempre de "usted"' in system_content


def test_default_tone_description_has_no_emoji_invitation():
    assert "emoji" not in DEFAULT_TONE_DESCRIPTION.lower()


@pytest.mark.asyncio
async def test_off_topic_message_conforms_to_floor():
    state = _make_state([], triage_decision="off_topic", user_text="clima")
    tenant_ctx = {"expertise": "salud", "tone_description": DEFAULT_TONE_DESCRIPTION, "contact_hint": ""}

    with patch("app.graph.nodes.generate._load_tenant", AsyncMock(return_value=tenant_ctx)):
        result = await generate(state)

    answer = result["answer"]
    assert "ayudarte" not in answer
    assert not any(ch in answer for ch in "😀😄🎉👋")


@pytest.mark.asyncio
async def test_greeting_message_conforms_to_floor():
    state = _make_state([], triage_decision="greeting", user_text="hola")
    tenant_ctx = {"expertise": "salud", "tone_description": DEFAULT_TONE_DESCRIPTION, "contact_hint": ""}

    with patch("app.graph.nodes.generate._load_tenant", AsyncMock(return_value=tenant_ctx)):
        result = await generate(state)

    answer = result["answer"]
    assert "ayudarte" not in answer
    assert not any(ch in answer for ch in "😀😄🎉👋")


# ---------------------------------------------------------------------------
# Compiled-graph seam — one case per triage decision
# ---------------------------------------------------------------------------

def _mock_chat_llm(reply_text: str, triage_decision: str):
    from app.schemas.triage import TriageDecision

    llm = MagicMock()
    llm.model_name = "test-model"
    structured = AsyncMock()
    structured.ainvoke = AsyncMock(return_value=TriageDecision(decision=triage_decision))
    llm.with_structured_output.return_value = structured
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply_text))
    return llm


def _mock_db_session(first_row):
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_result.first.return_value = first_row
    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    return db


def _initial_state(text: str, triage_decision: str = "rag") -> dict:
    return {
        "tenant_id": "acme",
        "thread_id": "tenant:acme:user:1:channel:telegram",
        "messages": [HumanMessage(content=text)],
        "retrieved_chunks": [],
        "triage_decision": triage_decision,
        "answer": "",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,user_text", [
    ("rag", "cuánto cuesta la biopsia"),
    ("catalog", "quiero el catálogo completo"),
])
async def test_compiled_graph_generation_prompt_carries_floor(decision, user_text):
    tenant_row = MagicMock(
        expertise_area="diagnóstico histológico",
        tone_description=DEFAULT_TONE_DESCRIPTION,
        contact_url=None,
        specialization_context="",
    )
    graph = build_graph(checkpointer=None)
    state = _initial_state(user_text, triage_decision=decision)
    llm = _mock_chat_llm("respuesta", triage_decision=decision)
    db = _mock_db_session(tenant_row)

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    system_prompt = llm.ainvoke.call_args[0][0][0].content
    assert _REGISTER_FLOOR in system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,user_text", [
    ("greeting", "hola"),
    ("off_topic", "quién ganó el partido"),
])
async def test_compiled_graph_static_reply_conforms_to_floor(decision, user_text):
    tenant_row = MagicMock(
        expertise_area="diagnóstico histológico",
        tone_description=DEFAULT_TONE_DESCRIPTION,
        contact_url=None,
        specialization_context="",
    )
    graph = build_graph(checkpointer=None)
    state = _initial_state(user_text, triage_decision=decision)
    llm = _mock_chat_llm("should not be used", triage_decision=decision)
    db = _mock_db_session(tenant_row)

    with (
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[])),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    answer = result["answer"]
    assert "ayudarte" not in answer
    assert not any(ch in answer for ch in "😀😄🎉👋")


# ---------------------------------------------------------------------------
# Compiled-graph seam — terser staff register (#27), one case per actor.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("is_staff", [True, False])
async def test_compiled_graph_register_variant_per_actor(is_staff):
    from app.graph.nodes.generate import _REGISTER_FLOOR, _REGISTER_FLOOR_STAFF

    tenant_row = MagicMock(
        expertise_area="diagnóstico histológico",
        tone_description=DEFAULT_TONE_DESCRIPTION,
        contact_url="https://acme.example/contact",
        specialization_context="",
    )
    graph = build_graph(checkpointer=None)
    state = _initial_state("cuánto cuesta un examen que no existe", triage_decision="rag")
    state["is_staff"] = is_staff
    llm = _mock_chat_llm("respuesta", triage_decision="rag")
    db = _mock_db_session(tenant_row)

    with (
        # Below exact_match_threshold but above handoff_threshold -> triggers
        # the negative-confirmation rule (contact/escalation line) without
        # crossing into an automatic escalation (see ADR-009 / #36), which
        # would suppress the contact hint for patients too.
        patch("app.graph.nodes.retrieve.retrieve_chunks", AsyncMock(return_value=[
            {"content": "x", "similarity": 0.5},
        ])),
        patch("app.graph.nodes.retrieve.rerank_chunks", AsyncMock(return_value=[{"content": "x", "similarity": 0.5}])),
        patch("app.graph.nodes.triage.get_triage_llm", return_value=llm),
        patch("app.graph.nodes.generate.get_chat_llm", return_value=llm),
        patch("app.graph.nodes.generate.AsyncSessionLocal", MagicMock(return_value=db)),
    ):
        await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    system_prompt = llm.ainvoke.call_args[0][0][0].content
    if is_staff:
        assert _REGISTER_FLOOR_STAFF in system_prompt
        assert "acme.example/contact" not in system_prompt
    else:
        assert _REGISTER_FLOOR in system_prompt
        assert "acme.example/contact" in system_prompt
