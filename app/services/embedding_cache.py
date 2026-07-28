import hashlib
import json
import logging
from typing import Protocol

from sqlalchemy import text

from app.config import settings
from app.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    async def aembed_query(self, text: str) -> list[float]: ...
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _cache_key(content: str) -> str:
    raw = f"{settings.embedding_model}:{settings.embedding_dim}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _get_cached(keys: list[str]) -> dict[str, list[float]]:
    if not keys:
        return {}
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT key, embedding FROM embedding_cache WHERE key = ANY(:keys)"),
            {"keys": keys},
        )
        # asyncpg has no vector codec registered for raw text() queries — the
        # column comes back as its literal "[0.1,0.2,...]" string form, not a
        # Python list, so it needs an explicit parse.
        return {row.key: json.loads(row.embedding) for row in result.fetchall()}


async def _store_cached(pairs: list[tuple[str, list[float]]]) -> None:
    if not pairs:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("""
                INSERT INTO embedding_cache (key, embedding)
                VALUES (:key, CAST(:vec AS vector))
                ON CONFLICT (key) DO NOTHING
            """),
            [{"key": k, "vec": str(v)} for k, v in pairs],
        )
        await db.commit()


class CachedEmbeddings:
    """Wraps an embedder with a content-addressed Postgres cache.

    Exposes only aembed_query/aembed_documents — the only two methods any
    caller in this codebase uses (rag.py, indexer.py) — so it's a drop-in
    replacement for the underlying LangChain embeddings object.
    """

    def __init__(self, underlying: _Embedder):
        self._underlying = underlying

    async def aembed_query(self, query: str) -> list[float]:
        vecs = await self.aembed_documents([query])
        return vecs[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if not settings.embedding_cache_enabled:
            return await self._underlying.aembed_documents(texts)

        keys = [_cache_key(t) for t in texts]
        cached = await _get_cached(keys)

        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            fresh = await self._underlying.aembed_documents(missing_texts)
            for i, vec in zip(missing_idx, fresh):
                cached[keys[i]] = vec
            await _store_cached([(keys[i], fresh[j]) for j, i in enumerate(missing_idx)])
            logger.info(
                "embedding_cache hits=%d misses=%d", len(texts) - len(missing_idx), len(missing_idx)
            )
        else:
            logger.info("embedding_cache hits=%d misses=0", len(texts))

        return [cached[k] for k in keys]
