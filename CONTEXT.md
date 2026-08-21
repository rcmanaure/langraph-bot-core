# Context

Glossary for this codebase. When naming a module, a test, an issue, or a proposal, use these terms — don't drift to synonyms.

Decisions live in [`docs/adr/`](docs/adr/). This file names things; ADRs record why.

## Tenant

One business served by the deployment. Owns a document corpus, per-channel credentials, and optional free-text configuration (`expertise_area`, `tone_description`, `specialization_context`). Identified everywhere by its **slug**, never its database id. Belongs to one [vertical](#vertical), which selects the prompt vocabulary it inherits.

One process serves all tenants — see [ADR-002](docs/adr/ADR-002-multi-tenant-thread-id.md).

## Vertical

The line of business a tenant is in: `medical_lab`, `gym`, `bakery`. One vertical serves many tenants — two labs share one — so the count grows with lines of business, never with clients.

It selects a [prompt pack](#prompt-pack) and nothing else. Distinct from `expertise_area`, which is a short label naming the business *to a person* (greetings, off-topic replies, the admin table); a vertical names the vocabulary *to the system*. See [ADR-011](docs/adr/ADR-011-prompt-packs.md).

## Prompt pack

The per-vertical vocabulary the prompts interpolate: triage examples, extraction examples, item-type names.

Vocabulary, never instructions. A pack colours what the model recognizes; it can never change how the model addresses anyone — the instruction skeleton and the register floor stay in code, out of reach of anything a tenant or operator can edit. That property is the whole point of [ADR-007](docs/adr/ADR-007-register-floor.md) and [ADR-011](docs/adr/ADR-011-prompt-packs.md) preserves it.

Sits underneath the per-tenant free text (`tone_description`, `specialization_context`), never replaces it.

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

## Canned answer

A tenant-authored reply to a question whose answer does not change: hours, location, payment methods, parking. Matched on operator-authored keyword sets before any model call and returned verbatim.

Never covers prices or availability. A canned answer is a copy of the truth with no expiry — stale hours are an annoyance, a stale price quoted to a patient is not. Anything with a number attached to a service stays [corpus](#chunk)-only.

## Not-offered term

An operator-declared keyword/synonym group naming something a tenant is known, with certainty, not to offer — edited one at a time or uploaded in bulk via Excel, per tenant. Matched before any model call, same shortcut position as a [canned answer](#canned-answer); a match produces a [not-offered denial](#not-offered-denial) at zero model/embedding cost.

Deliberately allowed to be incomplete. It is a fast path in front of the [not-offered denial](#not-offered-denial) mechanism, not a replacement for it — everything the list doesn't cover still falls through to that mechanism (or the normal turn) unchanged. See [ADR-012](docs/adr/ADR-012-not-offered-terms.md), including why this doesn't reopen the completeness problem [ADR-010](docs/adr/ADR-010-closed-world-denial.md) rejected.

## Not-offered denial

The reply a tenant gives when asked for a service it does not provide. Reached either deterministically, via a [not-offered term](#not-offered-term) match, or from the tenant's catalog treated as complete — absence from it means the tenant does not offer the thing — see [ADR-010](docs/adr/ADR-010-closed-world-denial.md).

Only a tenant marked closed-world can deny, and a denial needs two independent signals to agree; when they disagree, the thread [escalates](#escalation) instead.

Three things that are not the same: an escalation hands a conversation to a person, a refusal turns away a message that was never this business's to answer, and a denial answers the question that was asked with "we do not do that". A denial is recorded under its own name, never as an escalation.

## Human control

The state a thread is in between an escalation and its explicit return. While it holds, the bot is silent — it neither answers nor observes — and an [operator](#operator) answers in its place, for as many messages as they choose.

Control returns only when an operator ends it, or when an unclaimed escalation ages out. It returns silently, and the bot resumes without knowledge of what was said while it was held: that exchange is recorded for people to read, never folded back into the thread's checkpoint.

Distinct from the `human` [triage decision](#triage-decision), which classifies a single *message* as asking for a person. That decision is one of the things that causes an escalation; human control is the thread state that follows.

## Extraction

What the vision module reads out of an image: a literal item or procedure name, or several, or nothing it will vouch for. Never a guess — an unreadable image resolves to uncertainty and the user is asked to type instead.

The prompts were lab-specific — clinical vocabulary in triage, extraction and generation all assuming a diagnostic-lab tenant. [ADR-007](docs/adr/ADR-007-register-floor.md) records why that wasn't generalized alongside the register floor; [ADR-011](docs/adr/ADR-011-prompt-packs.md) is the generalization, moving that vocabulary into a [prompt pack](#prompt-pack) keyed by [vertical](#vertical).

## Chunk

An indexed slice of a tenant's corpus, embedded for retrieval. Stored in Postgres with pgvector — see [ADR-003](docs/adr/ADR-003-pgvector-storage.md).

## Triage decision

The classification the graph assigns an incoming message before doing any work: `greeting`, `catalog`, `rag`, `off_topic`, `human`, `canned`, `not_offered`. It decides which nodes run.

## Staff member

A channel-and-identifier pair an operator has nominated as staff for a tenant, in the admin panel — distinct from the person making an enquiry, who is never staff by default. Membership is a per-tenant, per-channel allowlist (`staff_members` table); resolving it reads only the channel-supplied identifier, never anything the person said. A message cannot claim staff status — see [ADR-006](docs/adr/ADR-006-staff-actor.md).

Being staff says how the bot answers *this* person, not who may answer *for* it: an escalated thread is answered by an [operator](#operator), who may or may not be a staff member anywhere.
