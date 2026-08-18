"""Add conversation_audit.chat_id and human_control_messages.author

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-18

Two columns for an operator to reply through the channel (#37, ADR-009):

- conversation_audit.chat_id: the channel's delivery target, captured once
  when a thread escalates (human_control.start()). Telegram's chat id
  differs from its user id and isn't derivable from thread_id, so sending
  outside a webhook needs it persisted somewhere.
- human_control_messages.author: the operator's free-text self-declared
  name, stamped on messages they send. Attribution only -- nothing verifies
  it (see CONTEXT.md's Operator entry). Null for "user" rows.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0015"
down_revision: Union[str, Sequence[str], None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("conversation_audit", sa.Column("chat_id", sa.String(100), nullable=True))
    op.add_column("human_control_messages", sa.Column("author", sa.String(128), nullable=True))


def downgrade() -> None:
    op.drop_column("human_control_messages", "author")
    op.drop_column("conversation_audit", "chat_id")
