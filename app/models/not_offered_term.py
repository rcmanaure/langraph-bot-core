from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, func

from app.models.base import Base


class NotOfferedTerm(Base):
    """Tenant-authored, keyword-triggered deterministic "we don't offer
    this" denial — matched in triage.py before any model call, same
    zero-cost shortcut position canned_answers already occupies (#53). No
    `answer` column: every match replies with the tenant's single
    `tenants.not_offered_message` (falling back to a vertical-neutral
    generic default if unset), never a per-term custom reply.

    ADR-010 rejected exactly this list shape as the SOLE mechanism for a
    closed-world denial ("unbounded and arbitrarily specific"). This table
    is not that: it's an opt-in, allowed-to-be-incomplete fast path that
    sits in front of ADR-010's mechanism, never a replacement for it. See
    ADR-012."""

    __tablename__ = "not_offered_terms"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    # list[str], normalized (app.services.text_normalize) at match time, not
    # at write time -- same rationale as CannedAnswer.keywords.
    keywords = Column(JSON, nullable=False)
    # "any" = any keyword present triggers the denial; "all" = every keyword
    # must be present.
    match_mode = Column(String(3), nullable=False, default="any", server_default="any")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
