"""Add human_control_messages table

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-17

The ordered message log the operator inbox reads from while a thread is
under human control -- distinct from conversation_audit, which is one row
per escalation, not per message. Purged on the same ninety-day schedule by
the same job (see app/scheduler.py).
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "human_control_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("thread_id", sa.String(255), nullable=False),
        sa.Column("sender", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_human_control_messages_thread_created",
        "human_control_messages", ["thread_id", "created_at"],
    )
    op.create_index(
        "ix_human_control_messages_created_at",
        "human_control_messages", ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_human_control_messages_created_at", table_name="human_control_messages")
    op.drop_index("ix_human_control_messages_thread_created", table_name="human_control_messages")
    op.drop_table("human_control_messages")
