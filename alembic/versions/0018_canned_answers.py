"""Add canned_answers table

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-20

Tenant-authored, keyword-triggered verbatim replies for static non-priced
questions (hours, location, payment methods) — see CONTEXT.md's "Canned
answer" and spec #44 (tickets #47/#50). ON DELETE CASCADE on tenant_id,
matching the convention 0002 established for every other tenant-scoped
table.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0018"
down_revision: Union[str, Sequence[str], None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "canned_answers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("match_mode", sa.String(length=3), nullable=False, server_default="any"),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_canned_answers_tenant_id", "canned_answers", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_canned_answers_tenant_id", table_name="canned_answers")
    op.drop_table("canned_answers")
