import logging
import os
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.channels.telegram import router as telegram_router
from app.channels.whatsapp import router as whatsapp_router
from app.config import settings
from app.middleware.security import add_security_middleware
from app.routes.admin import public_router as pricing_router
from app.routes.admin import router as admin_router
from app.routes.operator import router as operator_router

# LOG_LEVEL was set in docker-compose.dev.yml but never actually applied --
# nothing called basicConfig, so the root logger defaulted to WARNING and
# every INFO-level app log (generate_called, retrieve_top, etc.) was
# silently dropped, even in "dev" mode. Found live while trying to debug a
# real retrieval-quality report with no visibility into what actually ran.
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


async def _cleanup_stuck_jobs() -> None:
    """Delete partial chunks and mark RUNNING/PENDING jobs as FAILED on startup.

    Prevents partial embedding corruption if the process was killed mid-indexing.
    The job_id FK on document_chunks makes this a targeted DELETE, not a full scan.
    """
    from sqlalchemy import text

    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            text("SELECT id FROM index_jobs WHERE status IN ('RUNNING', 'PENDING')")
        )
        stuck = [str(r.id) for r in rows.fetchall()]
        if not stuck:
            return

        for job_id in stuck:
            await db.execute(
                text("DELETE FROM document_chunks WHERE job_id = :jid"),
                {"jid": job_id},
            )
        await db.execute(
            text("""
                UPDATE index_jobs
                   SET status = 'FAILED',
                       error_message = 'Startup cleanup: interrupted by server restart',
                       updated_at = now()
                 WHERE status IN ('RUNNING', 'PENDING')
            """)
        )
        await db.commit()
        logger.info("startup_cleanup_done stuck_jobs=%d", len(stuck))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.runtime import build_runtime
    from app.scheduler import start as start_scheduler
    from app.scheduler import stop as stop_scheduler

    await _cleanup_stuck_jobs()

    async with build_runtime() as runtime:
        app.state.graph = runtime.graph
        start_scheduler()
        logger.info("langgraph_ready")
        yield
        stop_scheduler()

    logger.info("shutdown_complete")


def _setup_sentry() -> None:
    if settings.sentry_dsn:
        sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.environment)


def _verify_redaction() -> None:
    """Hard boot-gate for exactly one failure mode: confirmed unmasked
    content in an exported span. Everything else (Phoenix unreachable,
    query timeout, ingest lag) is a soft failure identical in kind to
    _setup_phoenix's own soft-dependency contract -- register() succeeding
    does NOT mean Phoenix is actually reachable (OTel exporters are
    optimistic by design), so connectivity failures surface here instead,
    and must be treated the same way: log, don't trace, keep booting.

    Only the positive-confirmation case -- we reached Phoenix, found our own
    synthetic span, and the marker string is sitting in it unmasked -- is
    worth crashing boot over. This repo already lived through the
    alternative (dcac5f4: LOG_LEVEL silently not applied, discovered live in
    prod) -- a config flag with no enforcement is not a control, it's a
    hope. But "couldn't verify" must not be conflated with "verified and
    it's broken": the former can't distinguish a genuine redaction bug from
    Phoenix simply being down, and treating them the same would mean this
    self-check accidentally reintroduces the exact hard-boot-on-Phoenix-
    outage bug 1A exists to prevent.

    Creates one synthetic span with a known marker string in its input,
    force-flushes it, then queries Phoenix's own API for that span and
    asserts the marker is absent -- proving OPENINFERENCE_HIDE_* actually
    masks content rather than merely being set.
    """
    import time
    import uuid

    from opentelemetry import trace
    from phoenix.client import Client

    marker = f"REDACTION_SELFCHECK_{uuid.uuid4().hex}"
    tracer = trace.get_tracer("phoenix-redaction-selfcheck")
    with tracer.start_as_current_span(
        "redaction_selfcheck", openinference_span_kind="llm"
    ) as span:
        span.set_input(marker)
        span.set_output(marker)

    try:
        trace.get_tracer_provider().force_flush()
        time.sleep(1)  # let Phoenix finish ingesting the flushed span

        client = Client()
        spans = client.spans.get_spans(
            project_identifier="langraph-bot-v1",
            name="redaction_selfcheck",
            limit=5,
        )
    except Exception as e:
        # Could not reach Phoenix to verify -- soft failure, same contract
        # as _setup_phoenix. Do NOT hard-fail boot over connectivity.
        logger.warning(
            "phoenix_redaction_verify_unreachable error_type=%s error=%s",
            type(e).__name__,
            str(e),
        )
        return

    if not spans:
        # Reached Phoenix but our own span isn't there -- ingest lag or a
        # deeper wiring problem, not a confirmed content leak. Soft: log
        # loudly (this deserves attention) but don't crash boot on an
        # inconclusive result.
        logger.warning(
            "phoenix_redaction_verify_inconclusive "
            "reason=self_check_span_not_found_in_phoenix"
        )
        return

    for s in spans:
        if marker in str(s.attributes):
            raise RuntimeError(
                "Redaction self-check FAILED: marker string found in an "
                "exported span. OPENINFERENCE_HIDE_* env vars are not "
                "masking content. Refusing to boot with "
                "OBSERVABILITY_ENABLED=true -- fix redaction before enabling "
                "tracing against real traffic."
            )
    logger.info("phoenix_redaction_verified")


def _setup_phoenix() -> None:
    """Soft-dependency Phoenix tracing -- must never block app boot on its
    own account (unreachable, misconfigured, missing package).

    Redaction is mandatory whenever tracing runs, not configurable per-field:
    HIDE_INPUT_TEXT/HIDE_OUTPUT_TEXT/HIDE_INPUT_IMAGES are set unconditionally
    here rather than exposed as separate settings, so there's no toggle that
    could accidentally leave medical content unmasked.

    _verify_redaction() runs OUTSIDE the try/except below on purpose: once
    Phoenix is reachable and register() succeeds, a broken redaction check
    must hard-fail boot (see _verify_redaction's docstring), not be caught
    and swallowed like a Phoenix outage would be.
    """
    if not settings.observability_enabled:
        return

    # HIDE_INPUT_TEXT/HIDE_OUTPUT_TEXT alone do NOT cover it -- verified live
    # against Phoenix: those two only mask the structured
    # llm.input_messages.N.message.content attributes. The separate
    # input.value/output.value attributes (raw serialized request/response,
    # also populated on LLM-kind spans by LangChainInstrumentor) leaked the
    # full unmasked content until HIDE_INPUTS/HIDE_OUTPUTS were added here.
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"
    os.environ["OPENINFERENCE_HIDE_INPUT_TEXT"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUT_TEXT"] = "true"
    os.environ["OPENINFERENCE_HIDE_INPUT_IMAGES"] = "true"

    try:
        from phoenix.otel import register

        # batch=True: default is a SimpleSpanProcessor, which exports
        # synchronously on every span end -- every LLM call would block on
        # a network round-trip to Phoenix before returning, inflating the
        # exact p95 this instrumentation exists to measure, and turning a
        # Phoenix hang into a bot-response hang. Found by actually running
        # this against a stopped Phoenix and reading register()'s own
        # warning, not assumed.
        register(project_name="langraph-bot-v1", auto_instrument=True, batch=True)
        logger.info("phoenix_tracing_enabled")
    except Exception as e:
        logger.warning(
            "phoenix_setup_failed error_type=%s error=%s", type(e).__name__, str(e)
        )
        return

    _verify_redaction()


def create_app() -> FastAPI:
    _setup_sentry()
    _setup_phoenix()
    application = FastAPI(title="LangGraph RAG Bot", lifespan=lifespan)
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    add_security_middleware(application)
    return application


app = create_app()
app.include_router(pricing_router)  # Public: GET /pricing
app.include_router(operator_router)
app.include_router(admin_router)
app.include_router(telegram_router)
app.include_router(whatsapp_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
