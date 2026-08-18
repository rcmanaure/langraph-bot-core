from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text

from app.auth import verify_operator_key
from app.channels.base import Inbound
from app.channels.factory import build_adapter
from app.db import AsyncSessionLocal
from app.services import human_control
from app.services.redaction import redact_document_numbers
from app.services.security import validate_thread_id

_limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/operator", tags=["operator"])

# WhatsApp's customer service window -- a send outside it is rejected with a
# distinguishable reason rather than silently failing to deliver (#37).
_WA_WINDOW = timedelta(hours=24)


def _wa_window_remaining_seconds(channel: str, last_seen: datetime | None, now: datetime) -> int | None:
    """None when no window applies (not WhatsApp) or the user was never seen.
    Shared by list_pending and send_message so the 24h formula lives once."""
    if channel != "whatsapp" or last_seen is None:
        return None
    return max(0, int((_WA_WINDOW - (now - last_seen)).total_seconds()))


class ResumeRequest(BaseModel):
    text: str


@router.post("/resume/{thread_id}")
@_limiter.limit("20/minute")
async def resume(
    thread_id: str,
    body: ResumeRequest,
    request: Request,
    _: None = Depends(verify_operator_key),
):
    if not validate_thread_id(thread_id):
        raise HTTPException(status_code=422, detail="Invalid thread_id format")

    from langgraph.types import Command

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(Command(resume=body.text), config=config)

    # Mark interrupt resolved in audit
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                    UPDATE conversation_audit
                       SET expired_at = :now,
                           bot_response = :resp
                     WHERE thread_id = :tid
                       AND expired_at IS NULL
                       AND interrupt_started_at IS NOT NULL
                """),
                {
                    "now": datetime.now(timezone.utc),
                    "resp": body.text,
                    "tid": thread_id,
                },
            )
            await db.commit()
    except Exception as exc:
        # Non-fatal — audit failure must not block the operator response
        import logging
        logging.getLogger(__name__).warning("audit_update_failed thread=%s err=%s", thread_id, exc)

    answer = body.text
    if result and isinstance(result, dict):
        answer = result.get("answer") or answer

    return {"status": "resumed", "thread_id": thread_id, "answer": answer}


@router.get("/pending")
async def list_pending(
    _: None = Depends(verify_operator_key),
):
    """The listing an inbox renders from: every escalated thread across every
    active tenant, oldest waiting first, tenant named per row (#37). WhatsApp
    rows also report the remaining 24h service window, since a reply past it
    can't be delivered -- other channels get None, since no window applies.
    Excludes a deactivated tenant's threads -- a listed thread an operator
    can't actually reply to (send_message's build_adapter rejects inactive
    tenants) would be confusing to show at all."""
    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            text("""
                SELECT ca.thread_id, ca.user_id, ca.channel, ca.user_message,
                       ca.interrupt_started_at, t.slug AS tenant_slug,
                       wsw.last_user_message_at
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
    precedes every human_control_messages row for this thread."""
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

    for row in await human_control.thread_messages(thread_id):
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
