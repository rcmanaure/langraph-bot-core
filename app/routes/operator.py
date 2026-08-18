from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text

from app.auth import verify_operator_key
from app.channels.base import Inbound
from app.channels.factory import build_adapter
from app.db import AsyncSessionLocal
from app.services import human_control
from app.services.human_control import OPERATOR_MESSAGE_SINCE_ESCALATION_SQL
from app.services.redaction import redact_document_numbers
from app.services.security import validate_thread_id

_limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/operator", tags=["operator"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def operator_ui(request: Request):
    """The inbox page (#40) -- plain HTML/JS, no build step, served the same
    way admin.html already is (app/routes/admin.py's own /admin/ui)."""
    return templates.TemplateResponse(request=request, name="operator.html")

# WhatsApp's customer service window -- a send outside it is rejected with a
# distinguishable reason rather than silently failing to deliver (#37).
_WA_WINDOW = timedelta(hours=24)


def _wa_window_remaining_seconds(channel: str, last_seen: datetime | None, now: datetime) -> int | None:
    """None when no window applies (not WhatsApp) or the user was never seen.
    Shared by list_pending and send_message so the 24h formula lives once."""
    if channel != "whatsapp" or last_seen is None:
        return None
    return max(0, int((_WA_WINDOW - (now - last_seen)).total_seconds()))


@router.post("/resume/{thread_id}")
@_limiter.limit("20/minute")
async def resume(
    thread_id: str,
    request: Request,
    _: None = Depends(verify_operator_key),
):
    """The explicit end of human control (#39) -- no longer the operator's
    reply (that's POST /threads/{thread_id}/messages, #37). Two cases:
    a thread the reactive path (triage_decision == "human") suspended at
    interrupt_node is still genuinely blocked and must be unblocked with
    Command(resume=...) before the bot can answer again; a thread the
    proactive path (generate.py's own escalation) opened never suspended
    the graph at all, so there is nothing to resume -- only the audit row
    needs closing. graph.aget_state()'s `.next` distinguishes the two:
    non-empty means a node is still pending (genuinely suspended).

    ponytail: the check-then-act pair (aget_state, then ainvoke) races
    app/scheduler.py's own auto-expiry job, which resumes the same class of
    thread on its own 5-minute tick -- both sides check .next first, but
    nothing locks the row between the check and the resume. Same class of
    gap ADR-009 already accepts for two operators answering one thread at
    once ("a lock brings its own questions... a queue this size has not yet
    earned"); revisit together if either race actually bites in practice."""
    if not validate_thread_id(thread_id):
        raise HTTPException(status_code=422, detail="Invalid thread_id format")

    from langgraph.types import Command

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    if snapshot.next:
        # The resume value is discarded (see interrupt_node's own comment) --
        # per ADR-009, nothing said while a thread was held is folded back
        # into the graph's message history.
        await graph.ainvoke(Command(resume=None), config=config)

    await human_control.end(thread_id)

    return {"status": "resumed", "thread_id": thread_id}


@router.get("/pending")
@_limiter.limit("20/minute")
async def list_pending(
    request: Request,
    _: None = Depends(verify_operator_key),
):
    """The listing an inbox renders from: every escalated thread across every
    active tenant, oldest waiting first, tenant named per row (#37). WhatsApp
    rows also report the remaining 24h service window, since a reply past it
    can't be delivered -- other channels get None, since no window applies.
    Excludes a deactivated tenant's threads -- a listed thread an operator
    can't actually reply to (send_message's build_adapter rejects inactive
    tenants) would be confusing to show at all.

    Also reports who is attending, if anyone -- the author of the most
    recent operator message sent SINCE this escalation opened (same
    interrupt_started_at scoping app/scheduler.py's expiry predicate uses,
    for the same reason: thread_id is reused across escalations, so an
    unscoped lookup could attribute an old, unrelated reply to this one).
    Nothing stops two operators from both replying (#40, ADR-009) -- this
    only makes that visible, it doesn't prevent it.

    Rate-limited like every other /operator/* endpoint that mutates or
    authenticates -- operator.html's login form now uses this endpoint to
    verify a typed key (#40), which makes it a brute-force oracle for the
    shared operator secret if left unlimited."""
    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            text(f"""
                SELECT ca.thread_id, ca.user_id, ca.channel, ca.user_message,
                       ca.interrupt_started_at, t.slug AS tenant_slug,
                       wsw.last_user_message_at,
                       (SELECT hcm.author
                          FROM human_control_messages hcm
                         WHERE hcm.thread_id = ca.thread_id
                           AND {OPERATOR_MESSAGE_SINCE_ESCALATION_SQL}
                         ORDER BY hcm.created_at DESC
                         LIMIT 1) AS attending
                  FROM conversation_audit ca
                  JOIN tenants t ON t.id = ca.tenant_id
                  LEFT JOIN wa_service_windows wsw
                         ON wsw.tenant_id = ca.tenant_id AND wsw.user_id = ca.user_id
                 WHERE ca.expired_at IS NULL
                   AND ca.interrupt_started_at IS NOT NULL
                   AND t.active = true
                 ORDER BY ca.interrupt_started_at
            """)
        )
        now = datetime.now(timezone.utc)
        result = []
        for r in rows.fetchall():
            row = dict(r._mapping)
            last_seen = row.pop("last_user_message_at")
            row["wa_window_remaining_seconds"] = _wa_window_remaining_seconds(row["channel"], last_seen, now)
            result.append(row)
        return result


@router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: str,
    request: Request,
    _: None = Depends(verify_operator_key),
):
    """The full ordered conversation for one thread -- what the bot said
    before it gave up, plus whatever was exchanged since (#37). The bot's
    side lives in the LangGraph checkpoint (never written to
    conversation_audit, which is one row per escalation); the human-control
    side lives in human_control_messages. Concatenated, not interleaved by
    timestamp: the graph never runs again once a thread is under human
    control (see turn.py), so every checkpointed message necessarily
    precedes every human_control_messages row for this thread.

    The human-control side is scoped to the CURRENT open escalation (its
    interrupt_started_at) -- thread_id is reused across escalations, so an
    unscoped read would mix an old, already-resolved exchange into a brand
    new one (#40; same reason list_pending's "attending" lookup and
    app/scheduler.py's expiry predicate are scoped). No open escalation ->
    nothing current to show from that log."""
    if not validate_thread_id(thread_id):
        raise HTTPException(status_code=422, detail="Invalid thread_id format")

    messages: list[dict] = []
    graph = getattr(request.app.state, "graph", None)
    if graph is not None:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        for m in (snapshot.values or {}).get("messages", []):
            messages.append({
                "sender": "user" if m.type == "human" else "bot",
                "content": m.content if isinstance(m.content, str) else str(m.content),
                "author": None,
                "created_at": None,
            })

    async with AsyncSessionLocal() as db:
        escalation = (await db.execute(
            text("""
                SELECT interrupt_started_at FROM conversation_audit
                 WHERE thread_id = :thread
                   AND expired_at IS NULL
                   AND interrupt_started_at IS NOT NULL
            """),
            {"thread": thread_id},
        )).first()

    human_control_msgs = (
        await human_control.thread_messages(thread_id, since=escalation.interrupt_started_at)
        if escalation else []
    )
    for row in human_control_msgs:
        messages.append({
            "sender": row["sender"],
            "content": row["content"],
            "author": row["author"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        })

    return {"thread_id": thread_id, "messages": messages}


class SendMessageRequest(BaseModel):
    text: str
    author: str


@router.post("/threads/{thread_id}/messages")
@_limiter.limit("30/minute")
async def send_message(
    thread_id: str,
    body: SendMessageRequest,
    request: Request,
    _: None = Depends(verify_operator_key),
):
    """An operator reply, delivered straight through the channel -- the graph
    is never invoked (#37). `text` is masked before it's stored (same
    redaction the inbound turn applies) but delivered to the user unmasked;
    `author` is free-text attribution only, per CONTEXT.md's Operator entry."""
    if not validate_thread_id(thread_id):
        raise HTTPException(status_code=422, detail="Invalid thread_id format")

    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("""
                SELECT t.slug AS tenant_slug, ca.channel, ca.user_id, ca.chat_id,
                       wsw.last_user_message_at
                  FROM conversation_audit ca
                  JOIN tenants t ON t.id = ca.tenant_id
                  LEFT JOIN wa_service_windows wsw
                         ON wsw.tenant_id = ca.tenant_id AND wsw.user_id = ca.user_id
                 WHERE ca.thread_id = :thread
                   AND ca.expired_at IS NULL
                   AND ca.interrupt_started_at IS NOT NULL
                   AND t.active = true
            """),
            {"thread": thread_id},
        )).first()
    if not row:
        raise HTTPException(status_code=404, detail="No open escalation for this thread")

    if row.channel == "whatsapp":
        remaining = _wa_window_remaining_seconds(row.channel, row.last_user_message_at, datetime.now(timezone.utc))
        if not remaining:
            # Distinguishable from a generic failure -- an operator must never
            # believe a message was delivered when WhatsApp will simply drop it.
            raise HTTPException(status_code=409, detail={"error": "outside_service_window"})

    chat_id = row.chat_id
    if not chat_id:
        if row.channel == "whatsapp":
            chat_id = row.user_id  # equal by platform design, never just a guess
        else:
            # Escalated before chat_id existed (migration 0015) -- no way to
            # recover the real delivery target, and falling back to user_id
            # could silently misdeliver on the one channel this column exists
            # for (Telegram's chat id differs from its user id).
            raise HTTPException(status_code=409, detail={"error": "chat_id_unknown"})

    adapter = await build_adapter(row.tenant_slug, row.channel)
    if adapter is None:
        raise HTTPException(status_code=404, detail="Tenant not found or inactive")

    inbound = Inbound(tenant_slug=row.tenant_slug, channel=row.channel, user_id=row.user_id, chat_id=chat_id)
    delivered = await adapter.send(inbound, body.text)

    await human_control.record_message(
        row.tenant_slug, thread_id, "operator",
        redact_document_numbers(body.text), author=body.author,
    )
    # delivered=False means the channel adapter saw the send fail or skipped
    # it outright (e.g. missing credentials) -- still 200, since the message
    # IS recorded and the operator DID act, but an inbox must show this
    # distinctly from a normal send rather than implying success (#37).
    return {"status": "sent", "thread_id": thread_id, "delivered": delivered}
