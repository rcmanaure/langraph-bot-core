from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter

from app.config import settings
from app.services.embedding_cache import CachedEmbeddings

# max_retries=0 on all three factories below: ChatOpenRouter's own SDK-level
# retry/backoff (default max_retries=2) can add up to ~300s of hidden wall
# time on failure. We already retry/fall back ourselves (triage.py's
# structured+JSON-parse fallback, get_chat_llm(fallback=True)) -- stacking
# the SDK's own retry window on top would turn a fast failure into a
# multi-minute hang instead.

# lru_cache on every factory: constructing a ChatOpenRouter is NOT free.
# Because app_url is always set, its validate_environment() builds a fresh
# openrouter.OpenRouter SDK client backed by a NEW httpx.Client AND a new
# httpx.AsyncClient (see _build_client in langchain_openrouter/chat_models.py)
# -- each with its own connection pool, and nothing ever closes them. These
# factories are called per node per turn (triage, retrieve's rewrite,
# generate, update_profile), so every turn was leaking ~8 pooled clients and
# paying a fresh TCP+TLS handshake on every single LLM call instead of
# reusing a warm connection. Settings are read once at import and never
# mutated at runtime, so one instance per (factory, arg) is correct; the
# instances are stateless per request and safe to share across concurrent
# turns. Config changes need a restart -- already true of everything else in
# app.config.


@lru_cache(maxsize=None)
def get_chat_llm(fallback: bool = False) -> ChatOpenRouter:
    model = (settings.openai_fallback_model if fallback else None) or settings.openai_model
    return ChatOpenRouter(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        app_url=f"https://{settings.app_domain}",
        timeout=60000,
        max_retries=0,
    )


@lru_cache(maxsize=None)
def get_triage_llm() -> ChatOpenRouter:
    """Dedicated cheap/structured-output model for triage.py's classification
    call — split from get_chat_llm() (see ADR-008) since triage doesn't need
    generate.py's Spanish-fluency model. No fallback param: triage already
    degrades safely (JSON-parse retry, then defaults to "rag") without a
    model swap."""
    return ChatOpenRouter(
        model=settings.triage_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        app_url=f"https://{settings.app_domain}",
        timeout=60000,
        max_retries=0,
    )


@lru_cache(maxsize=None)
def get_vision_llm() -> ChatOpenRouter:
    return ChatOpenRouter(
        model=settings.openai_vision_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        app_url=f"https://{settings.app_domain}",
        timeout=60000,
        max_retries=0,
    )


def get_openrouter_headers() -> dict:
    """Auth + referer headers for raw HTTP calls to OpenRouter endpoints that
    aren't chat-completions (e.g. /rerank) and so can't go through ChatOpenRouter."""
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": f"https://{settings.app_domain}",
    }


@lru_cache(maxsize=None)
def get_embeddings() -> CachedEmbeddings:
    underlying = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        dimensions=settings.embedding_dim,
    )
    return CachedEmbeddings(underlying)
