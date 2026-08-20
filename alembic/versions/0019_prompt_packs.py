"""Add tenants.vertical and the prompt_packs table

Revision ID: 0019
Revises: 0018
Create Date: 2026-08-20

`tenants.vertical` selects a shared pack of prompt vocabulary instead of
each tenant carrying its own copy (see CONTEXT.md's "Vertical"/"Prompt
pack", ADR-011). server_default='medical_lab' both seeds sp-labs (today's
only tenant) and gives every future tenant a safe non-null default until
its real vertical is set.

prompt_packs is content, not instructions -- see ADR-011 for why only
vocabulary slots live here, never the instruction skeleton or the register
floor (those stay in code, see ADR-007).
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0019"
down_revision: Union[str, Sequence[str], None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("vertical", sa.String(length=50), nullable=True, server_default="medical_lab"),
    )
    op.create_table(
        "prompt_packs",
        sa.Column("vertical", sa.String(length=50), primary_key=True),
        sa.Column("rag_examples", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("prompt_packs")
    op.drop_column("tenants", "vertical")
