"""Add tenants.specialization_context free-text column

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-24

Free-text domain/jargon context fed into the RAG and vision prompts so the
model interprets vertical-specific terminology (e.g. medical/lab jargon).
Separate from expertise_area, which stays short — it doubles as a display
label in greeting/off-topic messages and the admin tenant table, so it can't
hold a long jargon paragraph without regressing those sites. Same
short-label-vs-long-content split as tone_description (see 0009).
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: Union[str, Sequence[str], None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("specialization_context", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("tenants", "specialization_context")
