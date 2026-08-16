# Context

Glossary for this codebase. When naming a module, a test, an issue, or a proposal, use these terms — don't drift to synonyms.

Decisions live in [`docs/adr/`](docs/adr/). This file names things; ADRs record why.

## Tenant

One business served by the deployment. Owns a document corpus, per-channel credentials, and optional free-text configuration (`expertise_area`, `tone_description`, `specialization_context`). Identified everywhere by its **slug**, never its database id.

One process serves all tenants — see [ADR-002](docs/adr/ADR-002-multi-tenant-thread-id.md).

## Channel

A messaging platform a tenant's users talk through: Telegram, WhatsApp. Not "integration", not "provider".

## Channel adapter

The module holding everything that varies between channels: credential verification, payload parsing, media fetching, and delivery. Everything that does *not* vary belongs to the inbound turn instead.

`ChannelAdapter` (`app/channels/base.py`) is the interface; `TelegramAdapter` and `WhatsAppAdapter` are the adapters that satisfy it.

## Inbound turn

One complete pass from a received message to a delivered reply: acknowledge, gate the media size, resolve the message to text (transcription or extraction), invoke the graph, send the answer.

The turn is channel-agnostic. When behaviour differs between Telegram and WhatsApp and the difference is not forced by the platform, that is a defect in the turn, not a feature of the channel.

## Inbound

The normalized result of parsing a webhook payload: who sent it, where to reply, and either text or a list of media refs. The only thing that crosses from a channel adapter into the inbound turn.

## Media ref

A pointer to media the channel holds — id, kind, size, mime type — resolvable to bytes through the channel adapter. Carries its size so the inbound turn can reject oversized media without downloading it.

## Thread

One conversation between one user and one tenant on one channel. The key is `tenant:{slug}:user:{id}:channel:{channel}`, and it doubles as the LangGraph checkpoint key — see [ADR-002](docs/adr/ADR-002-multi-tenant-thread-id.md).

Treat a thread id as sensitive: it grants access to a conversation's history.

## Extraction

What the vision module reads out of an image: a literal item or procedure name, or several, or nothing it will vouch for. Never a guess — an unreadable image resolves to uncertainty and the user is asked to type instead.

Deliberately vertical-agnostic. A medical order, a product label and a menu are all just documents with items on them.

## Chunk

An indexed slice of a tenant's corpus, embedded for retrieval. Stored in Postgres with pgvector — see [ADR-003](docs/adr/ADR-003-pgvector-storage.md).

## Triage decision

The classification the graph assigns an incoming message before doing any work: `greeting`, `catalog`, `rag`, `off_topic`. It decides which nodes run.

## Staff member

A channel-and-identifier pair an operator has nominated as staff for a tenant, in the admin panel — distinct from the person making an enquiry, who is never staff by default. Membership is a per-tenant, per-channel allowlist (`staff_members` table); resolving it reads only the channel-supplied identifier, never anything the person said. A message cannot claim staff status — see [ADR-006](docs/adr/ADR-006-staff-actor.md).
