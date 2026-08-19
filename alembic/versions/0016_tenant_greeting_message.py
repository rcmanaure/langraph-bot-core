"""Add tenants.greeting_message free-text column

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-19

Per-tenant override for the canned "greeting" triage reply, previously a
single hardcoded string in generate.py (_GREETING_MSG) that could only ever
suit one tenant. Nullable, no default -- NULL means "use the hardcoded
fallback" (see generate.py), so existing/new tenants that never set this
keep working unchanged. Same short-label-vs-long-content split rationale as
specialization_context (0010): this is greeting-specific, not folded into
tone_description, so an operator can set an exact literal reply (address,
phone, links) without it leaking into every other generated response.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0016"
down_revision: Union[str, Sequence[str], None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("greeting_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("tenants", "greeting_message")
