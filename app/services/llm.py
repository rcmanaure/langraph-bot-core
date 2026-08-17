from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings
from app.services.embedding_cache import CachedEmbeddings


def get_chat_llm(fallback: bool = False) -> ChatOpenAI:
    model = (settings.openai_fallback_model if fallback else None) or settings.openai_model
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={"HTTP-Referer": f"https://{settings.app_domain}"},
        timeout=60,
    )


def get_triage_llm() -> ChatOpenAI:
    """Dedicated cheap/structured-output model for triage.py's classification
    call — split from get_chat_llm() (see ADR-008) since triage doesn't need
    generate.py's Spanish-fluency model. No fallback param: triage already
    degrades safely (JSON-parse retry, then defaults to "rag") without a
    model swap."""
    return ChatOpenAI(
        model=settings.triage_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={"HTTP-Referer": f"https://{settings.app_domain}"},
        timeout=60,
    )


def get_vision_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_vision_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={"HTTP-Referer": f"https://{settings.app_domain}"},
        timeout=60,
    )


def get_openrouter_headers() -> dict:
    """Auth + referer headers for raw HTTP calls to OpenRouter endpoints that
    aren't chat-completions (e.g. /rerank) and so can't go through ChatOpenAI."""
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": f"https://{settings.app_domain}",
    }


def get_embeddings() -> CachedEmbeddings:
    underlying = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        dimensions=settings.embedding_dim,
    )
    return CachedEmbeddings(underlying)
