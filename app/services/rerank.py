import logging

import httpx
from opentelemetry import trace

from app.config import settings
from app.services.llm import get_openrouter_headers

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


async def _call_rerank_api(query: str, documents: list[str], top_n: int) -> list[dict]:
    # Named span: a raw httpx POST, not a LangChain runnable, so
    # OpenInference's auto-instrumentation never sees it -- without this it
    # shows up as unattributed dead time between retrieval and generation.
    with _tracer.start_as_current_span(
        "rerank", attributes={"openinference.span.kind": "RERANKER"}
    ):
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{settings.openrouter_base_url}/rerank",
                headers=get_openrouter_headers(),
                json={
                    "model": settings.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            response.raise_for_status()
            return response.json()["results"]


async def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-rank hybrid-search candidates with a cross-encoder before generation.

    Hybrid search (dense + keyword, RRF-fused) optimizes for recall — its
    fused score is a rank position blend, not a relevance judgment a
    dedicated reranker would agree with. This trades one extra HTTP round-trip
    for a shot at better precision on the final top_k that actually reaches
    the prompt. Any failure (rate limit, malformed response, disabled via
    config) falls back to the hybrid order as-is — reranking is a quality
    improvement, never a hard dependency for retrieval to work.
    """
    if not settings.rerank_enabled or len(chunks) <= top_k:
        return chunks[:top_k]

    documents = [c["content"][:300] for c in chunks]

    try:
        results = await _call_rerank_api(query, documents, top_k)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning("rerank_rate_limited falling back to hybrid order")
        else:
            logger.warning("rerank_failed=%s falling back to hybrid order", exc)
        return chunks[:top_k]
    except Exception as exc:
        logger.warning("rerank_failed=%s falling back to hybrid order", exc)
        return chunks[:top_k]

    ranked = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
    indices = [r["index"] for r in ranked]
    valid = [i for i in indices if 0 <= i < len(chunks)]
    if not valid:
        logger.warning("rerank_no_valid_indices falling back to hybrid order")
        return chunks[:top_k]

    reranked = [chunks[i] for i in valid[:top_k]]
    if len(reranked) < top_k:
        # API returned fewer relevant indices than top_k — backfill with
        # the remaining hybrid-order candidates rather than under-filling.
        seen = set(valid[:top_k])
        for i, c in enumerate(chunks):
            if len(reranked) >= top_k:
                break
            if i not in seen:
                reranked.append(c)
    return reranked
