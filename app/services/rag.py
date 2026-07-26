import logging

import tiktoken
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.llm import get_embeddings

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def token_counter(msgs) -> int:
    return sum(len(_enc.encode(m.content if isinstance(m.content, str) else "")) for m in msgs)


async def retrieve_chunks(db: AsyncSession, query: str, namespace: str) -> list[dict]:
    """Hybrid retrieval: dense (pgvector/HNSW) + keyword (tsvector), fused with
    Reciprocal Rank Fusion. Dense embeddings alone miss exact-match queries
    (item codes, exact names) that matter for a catalog bot — the keyword leg
    covers those; RRF combines both without needing to normalize/compare
    scores across the two different scales.
    """
    query_vec = await get_embeddings().aembed_query(query)

    ef = settings.hnsw_ef_search
    scan = settings.hnsw_iterative_scan
    if scan not in ("off", "relaxed_order", "strict_order"):
        scan = "relaxed_order"

    await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef)}"))
    await db.execute(text(f"SET LOCAL hnsw.iterative_scan = {scan}"))

    result = await db.execute(
        text("""
            WITH vector_search AS (
                SELECT id, content, source, page,
                       1 - (embedding <=> CAST(:qv AS vector)) AS similarity,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:qv AS vector)) AS rank
                FROM document_chunks
                WHERE namespace = :ns AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :cand_k
            ),
            text_search AS (
                SELECT id, content, source, page,
                       1 - (embedding <=> CAST(:qv AS vector)) AS similarity,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(content_tsv, websearch_to_tsquery('spanish', :q)) DESC
                       ) AS rank
                FROM document_chunks
                WHERE namespace = :ns
                  AND content_tsv @@ websearch_to_tsquery('spanish', :q)
                LIMIT :cand_k
            )
            SELECT
                COALESCE(v.content, t.content) AS content,
                COALESCE(v.source, t.source) AS source,
                COALESCE(v.page, t.page) AS page,
                COALESCE(v.similarity, t.similarity) AS similarity,
                COALESCE(1.0 / (:rrf_k + v.rank), 0.0)
                    + COALESCE(1.0 / (:rrf_k + t.rank), 0.0) AS fused_score
            FROM vector_search v
            FULL OUTER JOIN text_search t ON v.id = t.id
            ORDER BY fused_score DESC
            LIMIT :top_k
        """),
        {
            "qv": str(query_vec),
            "q": query,
            "ns": namespace,
            "cand_k": settings.hybrid_candidate_k,
            "rrf_k": settings.rrf_k,
            # Over-fetch beyond top_k_results when reranking is enabled — the
            # reranker needs a candidate pool to choose from, not just the
            # final count. Harmless overfetch when reranking is off.
            "top_k": max(settings.top_k_results, settings.rerank_candidate_k)
            if settings.rerank_enabled
            else settings.top_k_results,
        },
    )
    rows = result.fetchall()
    chunks = [
        {
            "content": r.content,
            "source": r.source,
            "page": r.page,
            "similarity": round(float(r.similarity), 3),
        }
        for r in rows
    ]
    if chunks:
        top = chunks[0]
        logger.info("retrieve_top ns=%s sim=%.3f src=%s", namespace, top["similarity"], top["source"])
    else:
        logger.warning("retrieve_empty ns=%s query_len=%d", namespace, len(query))
    return chunks


def cap_chunks_to_tokens(chunks: list[dict], max_tokens: int) -> list[dict]:
    kept, total = [], 0
    for chunk in chunks:
        n = len(_enc.encode(chunk["content"]))
        if total + n > max_tokens:
            break
        kept.append(chunk)
        total += n
    return kept
