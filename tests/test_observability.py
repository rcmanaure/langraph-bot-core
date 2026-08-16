"""Unit tests for Phoenix observability setup in app/main.py.
Mocked — no live services required. Live-Phoenix behavior (redaction
verification, real span production) is covered separately in
tests/test_observability_integration.py.

Tests cover:
  - OBSERVABILITY_ENABLED=false: register() never called
  - register() raises: app still boots, warning logged (soft dependency)
  - [CRITICAL regression] real boot with Phoenix unreachable: /health still
    responds -- 1A's contract must hold through the actual FastAPI app, not
    just the isolated function
  - register() succeeds: _verify_redaction() is called (wiring check)
"""
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_observability_disabled_skips_phoenix_setup():
    with (
        patch("app.main.settings.observability_enabled", False),
        patch("phoenix.otel.register") as mock_register,
    ):
        from app.main import _setup_phoenix

        _setup_phoenix()

    mock_register.assert_not_called()


@pytest.mark.asyncio
async def test_phoenix_register_raises_app_still_boots(caplog):
    with (
        patch("app.main.settings.observability_enabled", True),
        patch("phoenix.otel.register", side_effect=ConnectionError("phoenix unreachable")),
    ):
        from app.main import _setup_phoenix

        _setup_phoenix()  # must not raise

    assert "phoenix_setup_failed" in caplog.text
    assert "ConnectionError" in caplog.text


@pytest.mark.asyncio
async def test_phoenix_register_success_triggers_redaction_verify():
    """Wiring check: when register() succeeds, _verify_redaction() runs.
    Mocked here (no real Phoenix, no real SimpleSpanProcessor/
    BatchSpanProcessor construction, which would otherwise build real
    exporters and start a real background export thread) -- the real
    verification behavior is covered in test_observability_integration.py.
    """
    with (
        patch("app.main.settings.observability_enabled", True),
        patch("phoenix.otel.register", return_value=MagicMock()),
        patch("phoenix.otel.SimpleSpanProcessor", return_value=MagicMock()),
        patch("phoenix.otel.BatchSpanProcessor", return_value=MagicMock()),
        patch("app.main._verify_redaction") as mock_verify,
    ):
        from app.main import _setup_phoenix

        _setup_phoenix()

    mock_verify.assert_called_once()


@pytest.mark.asyncio
async def test_health_endpoint_survives_phoenix_unreachable():
    """CRITICAL regression: the real FastAPI app (not an isolated function
    call) must keep serving /health when Phoenix is unreachable and
    OBSERVABILITY_ENABLED=true. Reloads app.main under patched settings so
    the module-level `app = create_app()` + `/health` route registration
    both happen with Phoenix unreachable in effect -- `/health` is
    registered at module scope, after create_app() returns, so calling
    create_app() alone would not exercise the real route wiring."""
    with (
        patch("app.config.settings.observability_enabled", True),
        patch("phoenix.otel.register", side_effect=ConnectionError("phoenix unreachable")),
    ):
        import app.main as app_main

        importlib.reload(app_main)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app_main.app), base_url="http://test"
            ) as client:
                r = await client.get("/health")
            assert r.status_code == 200
            assert r.json() == {"status": "ok"}
        finally:
            # Restore the module to its normal (flag-off) state so later
            # tests in the same process don't inherit this reload.
            importlib.reload(app_main)
            del sys.modules["app.main"]
            importlib.import_module("app.main")
