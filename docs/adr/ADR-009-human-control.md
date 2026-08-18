# ADR-009: Human Control as a Thread State

**Status:** Accepted
**Date:** 2026-08

## Context

The bot could already be asked for a person: `triage` classifies such a message as `human`, `interrupt_node` suspends the graph, and `POST /operator/resume/{thread_id}` resumes it with the text a person typed. That covers one case — the user asking outright — and covers it for exactly one reply.

Three things were missing. The bot had no way to escalate on its own when it had said everything it could; there was nowhere to read the waiting conversations, only a JSON list at `GET /operator/pending`; and a person could not hold a conversation, only inject a single sentence into the bot's.

Two defects surfaced while designing this and are fixed as part of it. When the graph suspends, `answer` is empty and no node before `interrupt_node` has added a reply, so `run_turn`'s fallback echoes the user's own just-sent message back to them instead of saying someone is coming. And `_last_human_query` anchors a bare *affirmative* reply to the previous question but not a bare *negative* one, so "no" is embedded verbatim, retrieves unrelated chunks, and the bot answers a question nobody asked.

## Decision: a thread state, not a reply mode

An escalation moves a thread into human control, and it stays there until an operator ends it. This is the shape the rest of the decisions follow from.

### The bot answers, then escalates

An escalation does not replace the bot's reply — the bot says what it can in the same turn and the thread lands in the queue anyway. The user is not left holding a silence while waiting for a person, and the operator opens a thread where the bot has already said what it knew.

### Two escalation signals, neither of them the model's opinion

**A similarity floor.** `max(similarity)` across the retrieved pool below `handoff_threshold` (0.30) means nothing in the corpus is close to what was asked.

The floor deliberately reads the *maximum* across the pool, while the neighbouring `_has_confirmed_match()` deliberately reads `chunks[0]`. They fail in opposite directions and so must read different things: `_has_confirmed_match` guards against a *false confirmation*, so it must not let an unrelated-but-numerically-similar chunk vouch for a weak top match; the floor guards against a *false escalation*, so it must not fire while anything in the pool is close. Chunk order is RRF-fused and then cross-encoder reranked, so `chunks[0]` is frequently not the highest-similarity chunk — a keyword-exact hit inside a long price table can rank first on merit with a mediocre dense score, and thresholding on it would escalate a correct answer.

**Alternative considered and rejected:** escalate when `_has_confirmed_match()` is false, reusing the threshold that already exists. Rejected because that flag does not mean what the name suggests to a reader arriving from this feature. It selects `_MATCH_UNCONFIRMED_INSTRUCTION`, which offers the user an approximation and asks whether it is what they need — a productive path the user often accepts. Escalating there would route working conversations to a queue. The band that matters here, "definitivamente no hay nada relacionado", is named in the prompt text but had no number behind it; this ADR adds it.

**A rejected approximation.** `_REJECTION_RE` mirrors the existing `_CONFIRMATION_RE`: a whole-message negative, matched in `retrieve.py`, anchoring retrieval back to the previous query. It escalates only when the *previous* turn offered an approximation — a guard carried in state, without which every "no" escalates, including a "no" answering a confidently correct reply.

The regex earns its place on the retrieval bug alone. Escalation is the second thing it buys, and it is deliberate rather than accidental: with retrieval anchored, a rejection re-retrieves the same good chunks, the floor stays quiet, and only the explicit guard decides. Without the anchor, a bare "no" would have escalated by embedding to nothing and tripping the floor — the right outcome reached by an accident that would break the first time the embedding model changed.

**Alternative considered and rejected:** a new `triage` category for rejections, which would see full history and handle "no, yo buscaba otra cosa". Rejected for consistency with the pattern already in the file — regex for the unambiguous whole-message case, LLM for the rest, the same division `_GREETING_RE` makes in `triage.py`. The long-form rejection it misses carries content of its own and re-enters through normal retrieval, where the floor catches it if there is genuinely nothing.

**Catalog requests and staff never escalate.** Catalog chunks may be a raw dump with no similarity score at all, and a staff member is the business — there is nobody to hand them to but themselves. An empty chunk list does not escalate either: retrieval returns rows whenever the corpus holds one embedded chunk, so an empty list means the tenant was never indexed. That is an operational fault, and escalating on it would mirror the tenant's entire traffic into the queue.

The floor's value is chosen blind — no data existed to calibrate it. Every escalation logs the similarity that caused it, so the first weeks of real traffic are the calibration.

### The bot goes silent, and silence is enforced in one place

Under human control the bot sends nothing. Enforcing that per branch would have meant a guard at each of the thirteen points `turn.py` can speak from, including six inside `_resolve_text` that fire before the turn has any text to store — an unreadable image or a failed transcription would otherwise have the bot talking over a person mid-conversation.

Instead the real adapter is wrapped in a decorator whose `send` is a no-op and which delegates everything else. `ChannelAdapter` is a structural Protocol with no runtime checks, so the wrapper needs only the three methods the turn actually calls. One object, one place, and a seventh error message added later is silent by construction.

Media is still transcribed and extracted, so the operator reads text rather than a placeholder. The cost of a model call is accepted for a message the bot will not answer.

### Control is held, not borrowed

An operator sends as many messages as they want, delivered straight through the adapter without touching the graph, and ends control explicitly. The 30-minute expiry that already existed now applies only to *unclaimed* escalations: it exists so nobody waits forever for a person, not to disconnect a person who is already answering.

When control returns, it returns silently and the bot resumes with no knowledge of the exchange. Folding operator messages back into the checkpoint was considered and rejected: it means writing foreign turns into LangGraph's message history to improve a conversation a person has usually just resolved. The bot may re-ask something already settled. That is the accepted cost, and it is visible in the log rather than hidden in a merge.

### Attribution without an identity model

The operator key stays the only credential. An operator types a name, and it is stamped on each message they send.

**Alternative considered and rejected:** attribute to a `staff_members` row. Rejected because that table is keyed `(tenant, channel, identifier)` — a channel identity, existing so that someone *writing to the bot* gains privileges. Whoever answers the inbox may never write to the bot at all. It carries no name, so the picker would list phone numbers, and ADR-006 already accepts that one human spans several rows, so one person would attribute as several identities depending on which they picked. Since the design deliberately has no per-person authentication, a name that authenticates nothing does not need a table to live in.

## Consequences

**Positive:**

- The reactive path is fixed on the way past: asking for a person now produces a message saying one is coming, instead of the generic empty-answer fallback.
- `_REJECTION_RE` fixes garbage retrieval on a bare "no" — a defect that predates this feature and misfires on the bot's own most common follow-up question.
- `wa_service_windows` is finally read. It has been written on every inbound WhatsApp message and consumed by nothing; the remaining window is what tells an operator whether a reply can still be delivered.
- No new authentication, no identity table, no build toolchain: the inbox is static files served by the app that already exists, and `staff_members` and ADR-006 are untouched.

**Negative:**

- `handoff_threshold` is a guess until traffic calibrates it. Set too high it floods the queue; too low it never fires. The logging is what makes the mistake correctable rather than permanent.
- The bot loses the thread of a conversation a person handled, and may repeat a question already answered.
- Two operators can answer the same thread at once. Nothing prevents it; the inbox shows who is attending and coordination is left to the people. A lock brings its own questions — who releases an abandoned thread, and after how long — that a queue this size has not yet earned.
- A name that authenticates nothing can be typed as anyone's. Attribution here answers "who should I ask about this reply", not "who is accountable for it".
- An image the vision model cannot read reaches the inbox as the uncertainty text, not as the image. Showing media in the inbox is separate work.
- `/operator/*` now serves both senses the glossary separates. The routes are deployed and the name is kept; `CONTEXT.md` defines `Operator` to match what the code already means rather than renaming live endpoints to match a document.
