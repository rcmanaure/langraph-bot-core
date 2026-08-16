# ADR-005: Inbound Turn Design — Route-Driven Parsing and the Adapter Dedup Method

**Status:** Accepted
**Date:** 2026-08

## Context

The inbound turn — acknowledge the message, resolve it to text, invoke the graph, deliver the reply — was originally written once per channel. The two copies diverged: one channel's empty-answer fallback didn't exist on the other, a batched payload on one channel silently dropped every message after the first, and the two error strings differed by a sentence. None of this was a deliberate per-channel difference; it was drift, because there was no single implementation to keep in sync.

The fix was to extract one channel-agnostic inbound turn and put everything that genuinely varies per platform behind a channel adapter interface. That extraction raised two design questions that aren't obvious from reading the shipped code, and are recorded here so a future review doesn't re-propose them as improvements.

This record complements ADR-004 (per-channel webhook authentication): the adapter interface's verification method is what that decision named as the enforcement point, and this refactor is what finally made every adapter honour it consistently.

## Decision 1: Payload parsing is driven by the route, not the turn

The turn receives already-parsed inbound messages. It never sees a raw webhook payload and never calls an adapter's parsing method itself.

**Alternative considered and rejected:** hand the turn the raw payload and let the turn call the adapter's parse step internally. This was rejected because the route needs the parsed messages anyway, to compute a deduplication key per message before dispatching any background work — see Decision 2. If the turn owned parsing instead, it would run once per payload rather than once per message, reproducing exactly the batched-payload defect this refactor exists to fix: a payload carrying several user messages would again produce one combined turn (or only the first message's turn) instead of one independent turn per message.

Parsing therefore happens at the route, ahead of the turn, and produces a list of already-parsed messages — zero, one, or several — each of which becomes its own independent turn.

## Decision 2: The adapter interface carries a deduplication method

Deduplication looks at first glance like it belongs entirely behind the turn: it's a concern every channel needs, not something a specific platform forces. It isn't placed there.

The key is platform-specific: one platform's message identifier is already globally unique, while another's is only unique per sending account and needs a tenant-scoped prefix to avoid two different tenants' messages colliding on the same key. The cache that consumes the key, however, is not platform-specific — a single bounded, per-process cache of recently-seen keys serves every channel identically. So the adapter interface supplies the key-extraction method, and a cache shared across channels makes the keep-or-drop decision.

This also has a sequencing consequence: the dedup check has to happen before a turn is dispatched as background work, not inside the turn. A redelivered payload that dispatched its turn first and deduplicated second would already have produced the duplicate background work the check exists to prevent.

## Noted alongside: the media size gate has one platform exception

The turn owns the media size gate and applies it before fetching any media, using a size the parsed message already carries. One platform cannot supply that size at parse time — learning it requires a metadata round-trip against that platform's API, and the parsing step runs synchronously ahead of the webhook's fast response, so it cannot afford a network call. For that platform only, the gate is enforced one layer later, at the point the adapter actually fetches the media bytes, using the same rejection wording the turn's own gate would have used.

This is recorded explicitly as a platform exception, not a precedent: the turn is still the single owner of the gate's threshold and its user-facing wording. Only the timing of enforcement moves, and only for the one platform structurally unable to supply the size any earlier.

## Consequences

**Positive:**
- The divergences this refactor closed are no longer expressible. There is one inbound turn; a bug fix or a wording change made once reaches every channel, because there is nothing left to keep in sync.
- Adding a channel is bounded: it costs one adapter implementing the same six-method interface, not a second copy of the turn's branching logic.
- The turn's behaviour — transcription, extraction, size gating, batching, graph fallbacks — is tested once against an in-memory adapter instead of once per channel, so channels can no longer carry unequal test coverage for identical behaviour.

**Negative:**
- The adapter interface carries two responsibilities that are conceptually about the turn (deduplication, and — for one platform — the size gate) rather than being purely about platform mechanics. A reader has to know why before concluding they're misplaced.
- A future channel whose message identifiers arrive in a shape unlike either existing platform's has no third example to generalize from, and may need to extend the deduplication method's contract rather than merely implement it.
