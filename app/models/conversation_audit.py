import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base


class ConversationAudit(Base):
    """Write-once per turn. Used by admin UI and the interrupt expiry scheduler.
    LangGraph checkpoints store the full graph state; this table stores human-readable
    turn summaries and interrupt tracking metadata.
    """
    __tablename__ = "conversation_audit"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    thread_id = Column(String(255), nullable=False)
    user_id = Column(String(64), nullable=False)
    channel = Column(String(20), nullable=False)
    # The channel's delivery target, captured once at escalation
    # (human_control.start()) -- Telegram's chat id differs from its user id
    # and isn't derivable from thread_id, so an operator reply built outside
    # a webhook needs it persisted here (see ADR-009 / #37).
    chat_id = Column(String(100), nullable=True)
    user_message = Column(Text, nullable=True)
    bot_response = Column(Text, nullable=True)
    # Interrupt tracking — queried by expire_interrupted_threads scheduler
    interrupt_started_at = Column(DateTime(timezone=True), nullable=True)
    expired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_conversation_audit_thread", "thread_id"),
        Index("ix_conversation_audit_tenant_created", "tenant_id", "created_at"),
        # Serves the retention purge's created_at-only predicate — the
        # composite index above can't (created_at isn't its leftmost column).
        Index("ix_conversation_audit_created_at", "created_at"),
        # Partial index: scheduler query is O(interrupted rows) not O(all rows)
        Index(
            "ix_conversation_audit_interrupt_pending",
            "interrupt_started_at",
            postgresql_where=text("expired_at IS NULL AND interrupt_started_at IS NOT NULL"),
        ),
    )
