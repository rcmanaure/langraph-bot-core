from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, func

from app.models.base import Base


class NotOfferedDenial(Base):
    """One row per closed-world not-offered denial (#51/ADR-010) — a
    distinct outcome from an escalation or a normal answered turn, and the
    mechanism for discovering synonym gaps the two-signal rule accepts as
    its cost (see ADR-010's Consequences)."""

    __tablename__ = "not_offered_denials"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(255), nullable=False)
    query = Column(Text, nullable=False)
    max_similarity = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_not_offered_denials_tenant_created", "tenant_id", "created_at"),
    )
