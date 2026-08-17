"""Add standalone index on conversation_audit.created_at

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-16

The daily retention purge (app/scheduler.py:purge_old_conversation_audit)
filters on created_at alone. The existing composite index
(tenant_id, created_at) can't serve that predicate — created_at isn't its
leftmost column — so without this index the purge does a full table scan
every run instead of a range scan over the (small) set of expired rows.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_conversation_audit_created_at", "conversation_audit", ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_audit_created_at", table_name="conversation_audit")
