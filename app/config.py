from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://ragbot:ragbot@localhost:5432/ragbot"
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_checkpoint_pool_size: int = 5

    # Chat LLM — routed through OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openai_fallback_model: str = "deepseek/deepseek-v4-flash"

    # Embeddings — same key/base as chat; override only if using a different provider
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # RAG tuning
    chunk_size: int = 768
    chunk_overlap: int = 128
    top_k_results: int = 10
    history_max_tokens: int = 8000
    retrieval_max_tokens: int = 3000
    hnsw_ef_search: int = 160
    hnsw_iterative_scan: str = "relaxed_order"
    exact_match_threshold: float = 0.65
    # Hybrid search (dense + keyword, fused via RRF)
    hybrid_candidate_k: int = 30
    rrf_k: int = 60
    # Cross-encoder reranking of hybrid-search candidates before generation
    rerank_enabled: bool = True
    # Dev/testing escape hatches for every cache layer in the codebase --
    # vision_cache repeatedly confounded live debugging this session (a
    # photo re-tested minutes apart served a 2-day-old pre-fix result from
    # cache, hiding whether code fixes actually worked). All default True
    # (unchanged production behavior); set to false during active testing.
    vision_cache_enabled: bool = True
    embedding_cache_enabled: bool = True
    retrieve_cache_enabled: bool = True
    rerank_candidate_k: int = 20
    rerank_model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    # Channels
    telegram_bot_token: str = ""
    wa_phone_number_id: str = ""
    wa_verify_token: str = ""

    # Security
    secret_key: str = "changeme"
    fernet_key: str = ""
    csrf_secret: str = ""
    operator_token: str = ""  # if set, used for operator/admin auth instead of secret_key

    # Observability
    sentry_dsn: str = ""
    environment: str = "dev"
    # Phoenix (LLM tracing, replaces LangSmith -- open source, self-hosted
    # free, see docs/adr for the decision). Off by default -- flipped on
    # manually per environment only after the redaction self-check passes
    # (see _setup_phoenix in app/main.py).
    observability_enabled: bool = False

    # STT
    groq_api_key: str = ""
    stt_language: str = "es"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_domain: str = "localhost:8000"

    # Optional
    openai_vision_model: str = ""
    web_search_url: str = ""

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.openrouter_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openrouter_api_key


settings = Settings()

# Shared cap for every inbound media download (voice, audio, image), enforced
# by the inbound turn before any adapter fetches bytes. Lives here rather than
# in a feature module so neither the vision nor the STT path owns a limit the
# other also depends on.
MAX_MEDIA_BYTES = 10 * 1024 * 1024  # 10 MB

PLAN_LIMITS: dict[str, dict] = {
    "free":  {"docs": 5,   "chunks": 500,   "queries_monthly": 500},
    "basic": {"docs": 20,  "chunks": 2000,  "queries_monthly": 2000},
    "pro":   {"docs": 100, "chunks": 10000, "queries_monthly": 10000},
}
