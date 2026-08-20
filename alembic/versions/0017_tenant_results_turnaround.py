"""Add tenants.results_turnaround free-text column

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-20

Per-tenant override for the results-turnaround note appended to a priced
reply, previously a single hardcoded string in generate.py
(_RESULTS_NOTE_RULE, "Resultados: 3 a 5 días hábiles") that suited only one
tenant. Nullable, no default -- NULL means "omit the note", not an invented
generic turnaround, so a tenant without a meaningful one (or two labs with
different turnarounds) aren't forced into the same line. Same
short-label-vs-long-content split as greeting_message (0016).

Backfills sp-labs (the only tenant whose priced replies carried the
hardcoded note today) so this column's NULL default doesn't silently drop
that note from production replies the moment this ships -- found in
/code-review. Every other/future tenant stays NULL (no note) until an
operator sets one.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0017"
down_revision: Union[str, Sequence[str], None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("results_turnaround", sa.Text(), nullable=True))
    op.execute(
        "UPDATE tenants SET results_turnaround = '3 a 5 días hábiles' "
        "WHERE slug = 'sp-labs' AND results_turnaround IS NULL"
    )


def downgrade() -> None:
    op.drop_column("tenants", "results_turnaround")
