"""Add not_offered_terms table

Revision ID: 0023
Revises: 0022
Create Date: 2026-08-20

Per-tenant, deterministic not-offered term list (#53) — matched in
triage.py before any model call, same zero-cost shortcut position
canned_answers already occupies. No `answer` column: every match uses the
tenant's single `tenants.not_offered_message`. Column shape mirrors
canned_answers exactly minus `answer`. ON DELETE CASCADE on tenant_id,
matching the convention 0002 established for every other tenant-scoped
table.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0023"
down_revision: Union[str, Sequence[str], None] = "0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "not_offered_terms",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("match_mode", sa.String(length=3), nullable=False, server_default="any"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_not_offered_terms_tenant_id", "not_offered_terms", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_not_offered_terms_tenant_id", table_name="not_offered_terms")
    op.drop_table("not_offered_terms")
