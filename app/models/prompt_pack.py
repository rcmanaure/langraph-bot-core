from sqlalchemy import JSON, Column, DateTime, String, func

from app.models.base import Base


class PromptPack(Base):
    """Shared prompt vocabulary keyed by `tenants.vertical` — not a
    per-tenant copy. Vocabulary only, never instructions or register rules:
    the instruction skeleton and _REGISTER_FLOOR stay literal code in
    generate.py/triage.py, unreachable from any row here (see ADR-007,
    ADR-011, CONTEXT.md's "Prompt pack")."""

    __tablename__ = "prompt_packs"

    vertical = Column(String(50), primary_key=True)
    # Additional domain vocabulary appended to triage.py's fixed "rag"
    # examples -- list[str].
    rag_examples = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
