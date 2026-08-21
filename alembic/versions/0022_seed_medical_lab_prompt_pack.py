"""Seed the medical_lab prompt pack

Revision ID: 0022
Revises: 0021
Create Date: 2026-08-20

0019 created `prompt_packs` and backfilled `tenants.vertical = 'medical_lab'`
for sp-labs, but never inserted a matching pack row -- get_rag_examples()
(app/services/prompt_pack.py) has been returning [] for every tenant ever
since, silently. Found live: the admin UI's new Vertical dropdown (#48) had
nothing to offer besides sp-labs's own current value, because the table was
empty.

rag_examples here are the real study/procedure names from sp-labs's actual
indexed catalog (docs/sp-diagnostico-histologico-citologia-especifica.jsonl
and docs/sp-protocolos-oncologicos.jsonl -- the "name" field of every
record), not invented vocabulary. They're appended to _TRIAGE_PROMPT
(triage.py's _build_triage_prompt) as extra "rag" classification examples --
vocabulary only, per ADR-011, never an instruction.
"""

import json
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0022"
down_revision: Union[str, Sequence[str], None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Verbatim "name" field of every record in the two real sp-labs catalog
# source files (docs/sp-diagnostico-histologico-citologia-especifica.jsonl,
# docs/sp-protocolos-oncologicos.jsonl) -- not invented.
MEDICAL_LAB_RAG_EXAMPLES = [
    "Citología Exo-Endocervical",
    "Citología Endometrial",
    "Citología Vaginal",
    "Citología Vulvar",
    "Citología de Secreción Mamaria C/U",
    "Citología de Aspirado Mamario C/U",
    "LCR, Líquido Ascítico, Pleural, Abdominal – Bloque Celular",
    "Citología de Orina (Muestra Simple)",
    "Citología de Orina de 24 Horas",
    "Citología de Hisopado Rectal",
    "Citología de Hisopado Uretral",
    "Citología por PAAF de Tiroides C/U",
    "Citología por Punción de Ganglio Linfático",
    "Citología de Raspado Prepucial o Glande",
    "Citología de Esputo + Coloraciones Especiales",
    "Citología Lavado Bronquial + Coloraciones Especiales",
    "Cepillado Bronquial + Coloraciones Especiales",
    "Revisión de Láminas y Bloques",
    "Biopsia Extemporánea y Protocolo Ovario",
    "Biopsia Extemporánea y Protocolo Endometrio",
]


def upgrade() -> None:
    op.execute(
        sa.text(
            "INSERT INTO prompt_packs (vertical, rag_examples) "
            "VALUES ('medical_lab', CAST(:examples AS JSON)) "
            "ON CONFLICT (vertical) DO NOTHING"
        ).bindparams(examples=json.dumps(MEDICAL_LAB_RAG_EXAMPLES))
    )


def downgrade() -> None:
    op.execute("DELETE FROM prompt_packs WHERE vertical = 'medical_lab'")
