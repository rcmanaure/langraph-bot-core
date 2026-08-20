"""Add not_offered_denials table

Revision ID: 0021
Revises: 0020
Create Date: 2026-08-20

One row per closed-world not-offered denial (#51/ADR-010) — its own
outcome, distinct from an escalation or a normal answered turn, and the
mechanism for discovering synonym gaps the two-signal rule accepts as its
cost.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0021"
down_revision: Union[str, Sequence[str], None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "not_offered_denials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("max_similarity", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_not_offered_denials_tenant_created", "not_offered_denials", ["tenant_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_not_offered_denials_tenant_created", table_name="not_offered_denials")
    op.drop_table("not_offered_denials")
