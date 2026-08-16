import pytest
from langchain_core.messages import HumanMessage

from app.channels.telegram import _SEEN
from app.channels.whatsapp import _SEEN_WA


@pytest.fixture(autouse=True)
def clear_dedup_caches():
    _SEEN.clear()
    _SEEN_WA.clear()
    yield
    _SEEN.clear()
    _SEEN_WA.clear()


@pytest.fixture
def tenant_id() -> str:
    return "test-tenant"


@pytest.fixture
def thread_id(tenant_id: str) -> str:
    return f"tenant:{tenant_id}:user:12345:channel:telegram"


@pytest.fixture
def base_state(tenant_id: str, thread_id: str) -> dict:
    return {
        "tenant_id": tenant_id,
        "thread_id": thread_id,
        "messages": [HumanMessage(content="¿Cuál es el precio del servicio básico?")],
        "retrieved_chunks": [],
        "triage_decision": "rag",
        "answer": "",
    }


@pytest.fixture
def tenant_ctx() -> dict:
    return {"expertise": "servicios de consultoría", "contact_hint": ""}
