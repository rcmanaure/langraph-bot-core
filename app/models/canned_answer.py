from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, func

from app.models.base import Base


class CannedAnswer(Base):
    """Tenant-authored, keyword-triggered verbatim reply for a static
    non-priced question (hours, location, payment methods) — matched in
    triage.py before any model call (see CONTEXT.md's "Canned answer" and
    #47/#50). Never covers prices or availability; enforced by operator
    convention, not a runtime heuristic here (see ADR discussion in #44)."""

    __tablename__ = "canned_answers"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    # list[str], normalized (app.services.text_normalize) at match time, not
    # at write time -- so a normalizer improvement applies retroactively to
    # rows written before it, without a backfill.
    keywords = Column(JSON, nullable=False)
    # "any" = any keyword present triggers the reply; "all" = every keyword
    # must be present.
    match_mode = Column(String(3), nullable=False, default="any", server_default="any")
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
