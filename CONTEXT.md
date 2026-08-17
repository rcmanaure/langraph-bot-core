# Context

Glossary for this codebase. When naming a module, a test, an issue, or a proposal, use these terms — don't drift to synonyms.

Decisions live in [`docs/adr/`](docs/adr/). This file names things; ADRs record why.

## Tenant

One business served by the deployment. Owns a document corpus, per-channel credentials, and optional free-text configuration (`expertise_area`, `tone_description`, `specialization_context`). Identified everywhere by its **slug**, never its database id.

One process serves all tenants — see [ADR-002](docs/adr/ADR-002-multi-tenant-thread-id.md).

## Operator

Whoever holds the shared operator key. They administer the deployment — create tenants, upload a corpus, nominate staff members — and they answer threads under human control.

One credential, one term, no per-person identity: an operator self-declares a name when answering, and that name is attribution for people reading the log later, never a claim the system verifies. Distinct from a [staff member](#staff-member), whose identity *is* verified, comes from a channel, and grants a different thing entirely — see [ADR-006](docs/adr/ADR-006-staff-actor.md) and [ADR-009](docs/adr/ADR-009-human-control.md).

## Channel

A messaging platform a tenant's users talk through: Telegram, WhatsApp. Not "integration", not "provider".

## Channel adapter

The module holding everything that varies between channels: credential verification, payload parsing, media fetching, and delivery. Everything that does *not* vary belongs to the inbound turn instead.

`ChannelAdapter` (`app/channels/base.py`) is the interface; `TelegramAdapter` and `WhatsAppAdapter` are the adapters that satisfy it.

Not everything satisfying the interface is a channel. A decorator may wrap a real adapter to alter one behaviour for one turn — the silencing wrapper used under human control is one. Such a wrapper is named for what it changes, never for a platform.

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

A thread is answered by the bot by default. It can pass to human control, and pass back.

## Escalation

The moment a thread stops being the bot's to answer. Three things cause it: the user asks for a person outright, the retrieved corpus holds nothing close enough to what was asked, or the user rejects an approximation the bot offered.

An escalation is a decision about a *conversation*, computed in code from what retrieval returned — never a judgement the model narrates about itself. It is not the same as the bot declining to answer: an off-topic message is refused and stays the bot's.

## Human control

The state a thread is in between an escalation and its explicit return. While it holds, the bot is silent — it neither answers nor observes — and an [operator](#operator) answers in its place, for as many messages as they choose.

Control returns only when an operator ends it, or when an unclaimed escalation ages out. It returns silently, and the bot resumes without knowledge of what was said while it was held: that exchange is recorded for people to read, never folded back into the thread's checkpoint.

Distinct from the `human` [triage decision](#triage-decision), which classifies a single *message* as asking for a person. That decision is one of the things that causes an escalation; human control is the thread state that follows.

## Extraction

What the vision module reads out of an image: a literal item or procedure name, or several, or nothing it will vouch for. Never a guess — an unreadable image resolves to uncertainty and the user is asked to type instead.

The prompts are lab-specific today — clinical vocabulary in triage, extraction and generation all assume a diagnostic-lab tenant. See [ADR-007](docs/adr/ADR-007-register-floor.md) for why this wasn't generalized alongside the register floor. `TODOS.md`'s "Fold `tenant.specialization_context` into triage.py's classification prompt" item is the migration path once a second, differently-worded tenant is onboarded.

## Chunk

An indexed slice of a tenant's corpus, embedded for retrieval. Stored in Postgres with pgvector — see [ADR-003](docs/adr/ADR-003-pgvector-storage.md).

## Triage decision

The classification the graph assigns an incoming message before doing any work: `greeting`, `catalog`, `rag`, `off_topic`. It decides which nodes run.

## Staff member

A channel-and-identifier pair an operator has nominated as staff for a tenant, in the admin panel — distinct from the person making an enquiry, who is never staff by default. Membership is a per-tenant, per-channel allowlist (`staff_members` table); resolving it reads only the channel-supplied identifier, never anything the person said. A message cannot claim staff status — see [ADR-006](docs/adr/ADR-006-staff-actor.md).

Being staff says how the bot answers *this* person, not who may answer *for* it: an escalated thread is answered by an [operator](#operator), who may or may not be a staff member anywhere.
