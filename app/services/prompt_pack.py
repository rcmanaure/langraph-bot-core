"""TTL-cached loading of a vertical's prompt pack — vocabulary only (see
CONTEXT.md's "Prompt pack", ADR-011). No write path exists yet (packs are
developer-authored, not operator-authored), so the TTL alone is enough;
mirrors app/services/canned.py's cache shape for consistency.
"""

import time

from sqlalchemy import text

from app.db import AsyncSessionLocal

_TTL_SECONDS = 90

# vertical -> (cached_at, rag_examples)
_cache: dict[str, tuple[float, list[str]]] = {}


async def get_rag_examples(vertical: str | None) -> list[str]:
    """Extra domain vocabulary for triage.py's "rag" category — never
    instructions, never a register rule (see ADR-007/ADR-011). Empty for a
    tenant with no vertical set, so an unconfigured tenant's prompt is
    byte-identical to before this feature."""
    if not vertical:
        return []

    now = time.monotonic()
    cached = _cache.get(vertical)
    if cached and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("SELECT rag_examples FROM prompt_packs WHERE vertical = :v"),
            {"v": vertical},
        )).first()

    examples = list(row.rag_examples) if row and row.rag_examples else []
    _cache[vertical] = (now, examples)
    return examples
