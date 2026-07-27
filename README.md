# LangGraph RAG Bot

Multi-tenant conversational RAG bot for Telegram (and WhatsApp) built on LangGraph. Each tenant has its own knowledge base, bot token, and expertise area. Users ask questions in natural language; the bot retrieves relevant chunks from the tenant's indexed documents and answers using an LLM.

---

## What it does

- Answers questions from indexed documents using semantic search (pgvector) + LLM generation
- Classifies every message: `rag` (document lookup), `catalog` (full list), `human` (escalate to operator), `off_topic` (decline politely)
- Maintains conversation history per user via LangGraph checkpoints (PostgreSQL)
- Supports voice messages (transcribed via Groq Whisper)
- Supports photo messages on Telegram — extracts the procedure/item being asked about via a vision model, then answers from the catalog
- Human-in-the-loop: operator can take over any conversation thread
- Admin panel at `/admin/ui` to manage tenants and upload knowledge base documents

---

## Stack

| Layer | Tech |
|---|---|
| Graph / RAG | LangGraph 0.3+, LangChain Core |
| LLM + embeddings | OpenRouter (any model; default `openrouter/free`, embeddings `openai/text-embedding-3-small`) |
| Database | PostgreSQL 16 + pgvector 0.8 |
| Checkpoints | LangGraph `AsyncPostgresSaver` |
| API | FastAPI + Uvicorn |
| Admin UI | Single-page HTML, Alpine.js, Tailwind CDN |
| Channels | Telegram Bot API, WhatsApp Cloud API (optional) |
| STT | Groq Whisper (optional, for voice messages) |
| Observability | LangSmith |
| Infra | Docker Compose + Traefik (production), cloudflared (local dev) |
| Package manager | uv |

---

## LangGraph flow

```
validate → retrieve → triage ──► generate → validate_output → respond
                              └──► interrupt_node  (human escalation)
```

- **validate** — injection scan, message trimming
- **retrieve** — pgvector similarity search against tenant namespace
- **triage** — LLM classifies intent: `rag` / `catalog` / `human` / `off_topic`
- **generate** — LLM answers using retrieved chunks (or full catalog)
- **validate_output** — safety check on generated answer
- **respond** — sends final message back to channel
- **interrupt_node** — pauses thread; operator resumes via `/operator/resume`

---

## Quick start (local)

### Prerequisites

- Docker + Docker Compose
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- An [OpenRouter](https://openrouter.ai) API key

### 1. Clone and configure

```bash
git clone https://github.com/rcmanaure/langraph-bot-v1.git
cd langraph-bot-v1
cp .env.example .env
# Edit .env — fill in OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, SECRET_KEY
```

### 2. Start services

```bash
# Create the shared Docker network (one-time)
docker network create lgbot-net

docker compose up -d
```

The API starts at `http://localhost:8000`. Migrations run automatically on startup.

### 3. Expose a public webhook (local dev)

Telegram requires a public HTTPS URL. On Windows, `scripts\start-tunnel.ps1` starts cloudflared **and** registers the webhook automatically:

```powershell
# Fill in TELEGRAM_BOT_TOKEN, WEBHOOK_SECRET, and TENANT_SLUG in .env first
.\scripts\start-tunnel.ps1
# Starts the tunnel, waits for the trycloudflare.com URL, calls setWebhook — all in one step.
```

On Linux/macOS, start cloudflared manually and then follow step 5 below to register the webhook.

### 4. Create a tenant

Open the admin panel and log in with your operator key:

```
http://localhost:8000/admin/ui
```

The operator key is `SHA256(SECRET_KEY)`. Generate it:

```bash
echo -n "your-SECRET_KEY-value" | sha256sum
```

In the **Tenants** tab, fill in slug, bot token, webhook secret, and expertise area.

### 5. Register the Telegram webhook (Linux/macOS)

```bash
# After starting cloudflared and getting the URL:
curl -X POST "https://api.telegram.org/bot<BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://<tunnel>.trycloudflare.com/webhook/telegram/<tenant_slug>",
    "secret_token": "<webhook_secret>"
  }'
```

> **Note:** `scripts\start-tunnel.ps1` (Windows) does steps 3 + 5 together.

### 6. Index a document

In the admin panel, go to the **Documentos** tab, select your tenant, drag in a PDF, `.md`, or `.jsonl` catalog file, and click **Subir e indexar**. A progress bar tracks chunking and embedding. Re-uploading a file with the same name replaces its previous chunks automatically. See `docs/catalog-schema.jsonl` for the JSONL catalog format (price lists, per-item keywords).

The bot is now live — message it on Telegram.

---

## Admin panel

`GET /admin/ui` — no auth required to load the page; login uses the operator key.

| Tab | What it does |
|---|---|
| Tenants | List all tenants, create new ones (returns a one-time API key) |
| Documentos | Upload PDF, Markdown, or JSONL catalog; track indexing progress in real time |
| Jobs | History of all indexing jobs across tenants |

---

## API reference

### Webhook

| Method | Path | Description |
|---|---|---|
| `POST` | `/webhook/telegram/{tenant_slug}` | Telegram update receiver |
| `GET` | `/webhook/whatsapp/{tenant_slug}` | WhatsApp webhook verification |
| `POST` | `/webhook/whatsapp/{tenant_slug}` | WhatsApp Cloud API receiver |

### Admin (requires `X-Operator-Key` header)

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/ui` | Admin panel HTML (no auth) |
| `GET` | `/admin/tenants` | List tenants |
| `POST` | `/admin/tenants` | Create tenant (returns one-time API key) |
| `PATCH` | `/admin/tenants/{slug}` | Update tenant fields |
| `DELETE` | `/admin/tenants/{slug}` | Delete tenant |
| `GET` | `/admin/tenants/{slug}/webhook-status` | Check Telegram webhook registration status |
| `POST` | `/admin/tenants/{slug}/regen-key` | Rotate tenant API key |
| `GET` | `/admin/billing/{tenant_slug}` | Tenant billing/plan info |
| `POST` | `/admin/index` | Upload + index document |
| `GET` | `/admin/index/{job_id}` | Job status |
| `GET` | `/admin/index?tenant_slug=X` | List jobs for tenant |

### Operator (requires operator token)

| Method | Path | Description |
|---|---|---|
| `POST` | `/operator/resume/{thread_id}` | Resume a human-escalated conversation |
| `GET` | `/operator/pending` | List threads awaiting operator response |

### Health

```
GET /health  →  {"status": "ok"}
```

---

## Authentication

**Operator key** — used by the admin panel and `/admin/*` routes:

```
X-Operator-Key: sha256(SECRET_KEY)
```

**Tenant API key** — generated on tenant creation (shown once). Used for future per-tenant integrations.

**Telegram webhook secret** — set per tenant in the DB; Telegram sends it as `X-Telegram-Bot-Api-Secret-Token` on every update.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string (`postgresql+asyncpg://...`) |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `OPENAI_MODEL` | No | Chat model (default: `openrouter/free`) |
| `OPENAI_FALLBACK_MODEL` | No | Fallback if primary fails |
| `EMBEDDING_MODEL` | No | Embedding model (default: `openai/text-embedding-3-small`) |
| `SECRET_KEY` | Yes | Used to derive the operator key |
| `FERNET_KEY` | Yes | Fernet key for encrypting WhatsApp tokens at rest |
| `TELEGRAM_BOT_TOKEN` | No | Global fallback bot token (per-tenant overrides this) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key for tracing |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `GROQ_API_KEY` | No | Groq API key for voice transcription |
| `OPENAI_VISION_MODEL` | No | Vision model for photo messages on Telegram (unset disables photo support) |
| `TRAEFIK_HOST` | No | Production domain (e.g. `bot.yourdomain.com`) |
| `SENTRY_DSN` | No | Sentry error tracking |

Full list with descriptions: see `.env.example`.

---

## Development

```bash
# Install dependencies
uv sync

# Run locally (without Docker)
DATABASE_URL=postgresql+asyncpg://ragbot:ragbot@localhost:5432/ragbot uv run uvicorn app.main:app --reload

# Run migrations
uv run alembic upgrade head

# Run tests (excludes slow eval tests)
uv run pytest -m "not eval" --tb=short -q
```

### Test suite

| File | What it covers |
|---|---|
| `test_telegram_webhook.py` | Webhook handler edge cases (auth, routing, voice, graph errors) — no live services |
| `test_whatsapp_dedup.py` | WhatsApp dedup cache — LRU eviction, duplicate message detection |
| `test_tenant_crud.py` | Tenant CRUD endpoints (create, patch, delete, regen-key, webhook-status) — no live services |
| `test_admin_api.py` | Admin API edge cases (auth, tenant CRUD, indexing jobs) — no live services |
| `test_catalog_qa.py` | Catalog response QA — verifies correct context/chunks passed to LLM |
| `test_graph.py` | LangGraph routing logic |
| `test_nodes.py` | Individual node behavior (triage fallback paths, fence stripping) |
| `test_policies.py` | Plan-based policy enforcement (free/basic/pro limits) |
| `test_indexing.py` | Document chunking + embedding pipeline |
| `test_security.py` | Injection scanner, rate limiting |
| `test_scheduler.py` | APScheduler interrupt expiry |
| `test_evals.py` | LLM quality evals (slow, requires API keys) |

---

## Production deployment

The Docker Compose file includes Traefik labels for automatic HTTPS via Let's Encrypt:

```bash
# Set TRAEFIK_HOST in .env
echo "TRAEFIK_HOST=bot.yourdomain.com" >> .env

docker compose up -d
```

Traefik must be running on the `lgbot-net` Docker network with a `letsencrypt` certificate resolver configured. Register the Telegram webhook pointing to `https://bot.yourdomain.com/webhook/telegram/<slug>`.

---

## Project structure

```
app/
├── channels/
│   ├── base.py      # ChannelEvent dataclass + ChannelAdapter Protocol
│   ├── telegram.py  # TelegramAdapter + webhook handler
│   └── whatsapp.py  # WhatsAppAdapter + webhook handler
├── graph/
│   ├── builder.py   # LangGraph StateGraph definition
│   └── nodes/       # validate, retrieve, triage, generate, validate_output
├── middleware/      # Security headers, request size limit
├── models/          # SQLAlchemy ORM models
├── routes/          # admin.py, operator.py
├── services/        # indexer, rag, llm, stt
├── templates/       # admin.html (admin panel)
├── policies.py      # TenantPolicy + PolicyEngine (policy-as-code)
├── state.py         # AgentState TypedDict (versioned schema contract)
└── main.py          # FastAPI app + lifespan
alembic/             # Database migrations
docs/
├── adr/                 # Architecture Decision Records (ADR-001 … ADR-004)
├── agent-dna.md         # AgentState field-by-field contract and versioning rules
└── catalog-schema.jsonl # JSONL catalog format reference (fields, examples)
tests/               # pytest test suite
CHANGELOG.md         # Release history
DESIGN.md            # Admin panel design system (colors, typography, components)
TODOS.md             # Accepted deferred work items
VERSION              # Current version (semver: major.minor.patch.build)
scripts/start-tunnel.ps1  # Windows: starts cloudflared tunnel + registers Telegram webhook
scripts/benchmark_accuracy.py  # Vision/OCR extraction accuracy benchmark
```
