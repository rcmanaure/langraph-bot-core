from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func

from app.models.base import Base


class StaffMember(Base):
    """A channel-and-identifier pair an operator has nominated as staff for a
    tenant. Identity comes from the channel (who actually sent the message),
    never from message text — see ADR-006."""
    __tablename__ = "staff_members"
    __table_args__ = (
        UniqueConstraint("tenant_id", "channel", "identifier", name="uq_staff_member_tenant_channel_identifier"),
    )

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    channel = Column(String(32), nullable=False)
    identifier = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
