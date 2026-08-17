"""Add staff_members table, drop unused tenants.operator_chat_id

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-16

staff_members is the per-tenant, per-channel allowlist backing staff
resolution in the inbound turn (see ADR-006) — a (tenant_id, channel,
identifier) triple, never a claim read out of message text.

tenants.operator_chat_id was added in the baseline migration and never read
by any code path; dropped as dead schema now that staff identity has a real
home.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "staff_members",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column("identifier", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "channel", "identifier",
                             name="uq_staff_member_tenant_channel_identifier"),
    )
    op.create_index("ix_staff_members_tenant_id", "staff_members", ["tenant_id"])
    op.drop_column("tenants", "operator_chat_id")


def downgrade() -> None:
    op.add_column("tenants", sa.Column("operator_chat_id", sa.String(64), nullable=True))
    op.drop_index("ix_staff_members_tenant_id", table_name="staff_members")
    op.drop_table("staff_members")
