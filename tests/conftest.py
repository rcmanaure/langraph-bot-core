from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.channels import telegram, whatsapp


@pytest.fixture(autouse=True)
def clear_dedup_caches():
    telegram._SEEN.clear()
    whatsapp._SEEN.clear()
    yield
    telegram._SEEN.clear()
    whatsapp._SEEN.clear()


@pytest.fixture(autouse=True)
def mock_resolve_staff():
    """The inbound turn resolves staff membership via a real DB query
    (app.services.staff.resolve_staff) before every graph invocation.
    Default every test to non-staff so tests that don't care about staff
    resolution don't need their own DB mock; tests that do care (see
    tests/test_turn.py) inspect or override this fixture's mock."""
    with patch("app.channels.turn.resolve_staff", new_callable=AsyncMock, return_value=False) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_under_human_control():
    """Same rationale as mock_resolve_staff above -- app.channels.turn checks
    a real DB-backed predicate (is_under_human_control) before every graph
    invocation. Default every test to "not under human control" so tests
    that don't care don't need their own DB mock; tests/test_turn.py's
    under_human_control fixture overrides this for the cases that do."""
    with patch("app.channels.turn.is_under_human_control", new_callable=AsyncMock, return_value=False) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_triage_tenant_lookups():
    """triage() unconditionally checks canned answers, not-offered terms, and
    the tenant's prompt-pack vertical before calling the LLM (#47/#48/#53).
    Same rationale as mock_resolve_staff above: default every test to "no
    canned match, no not-offered match, no vertical" so tests that don't
    care don't need their own DB mock; tests that do care
    (tests/test_nodes.py) patch these targets themselves, which overrides
    this default within their own `with patch(...)` block."""
    with (
        patch("app.graph.nodes.triage.match_canned_answer", new_callable=AsyncMock, return_value=None),
        patch("app.graph.nodes.triage.match_not_offered_term", new_callable=AsyncMock, return_value=False),
        patch("app.graph.nodes.triage.get_tenant_vertical", new_callable=AsyncMock, return_value=None),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_retrieve_tenant_lookups():
    """retrieve() unconditionally looks up specialization/expansion context
    and the closed-world catalog flag (#49) via asyncio.gather. Same
    rationale as mock_resolve_staff above; tests/test_retrieve_node.py
    patches these targets itself for the cases that care."""
    with (
        patch("app.graph.nodes.retrieve.get_tenant_specialization", new_callable=AsyncMock, return_value=""),
        patch(
            "app.graph.nodes.retrieve.get_tenant_closed_world_context",
            new_callable=AsyncMock,
            return_value={"expertise_area": "", "catalog_is_closed": False},
        ),
    ):
        yield


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
