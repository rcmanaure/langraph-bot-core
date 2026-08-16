"""Drop conversation_audit.langsmith_trace_url

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-16

Dead column -- nothing writes it since the switch from LangSmith to
Phoenix for tracing. Phoenix trace URLs aren't per-turn (spans are queried
by project/time in the Phoenix UI), so there's no replacement value to add.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0013"
down_revision: Union[str, Sequence[str], None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("conversation_audit", "langsmith_trace_url")


def downgrade() -> None:
    op.add_column(
        "conversation_audit",
        sa.Column("langsmith_trace_url", sa.String(512), nullable=True),
    )
