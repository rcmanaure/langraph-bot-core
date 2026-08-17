from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text, func

from app.models.base import Base


class HumanControlMessage(Base):
    """One row per message recorded while a thread is under human control --
    the ordered log the operator inbox reads from. Distinct from
    ConversationAudit, which is one row per escalation, not per message.
    """
    __tablename__ = "human_control_messages"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(255), nullable=False)
    sender = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_human_control_messages_thread_created", "thread_id", "created_at"),
        # Serves the retention purge's created_at-only predicate — the
        # composite index above can't (created_at isn't its leftmost column).
        Index("ix_human_control_messages_created_at", "created_at"),
    )
