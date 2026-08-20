"""Add tenants.catalog_is_closed and tenants.not_offered_message

Revision ID: 0020
Revises: 0019
Create Date: 2026-08-20

Plumbing for the closed-world not-offered denial (#49, ADR-010). Opt-in,
default false — a tenant that never sets catalog_is_closed is unaffected.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0020"
down_revision: Union[str, Sequence[str], None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("catalog_is_closed", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column("tenants", sa.Column("not_offered_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("tenants", "not_offered_message")
    op.drop_column("tenants", "catalog_is_closed")
