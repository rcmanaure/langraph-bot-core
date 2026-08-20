# ADR-010: The Catalog as a Closed World

**Status:** Accepted
**Date:** 2026-08

## Context

A tenant that does not perform a study still gets asked about it. Today that question retrieves nothing close, `generate()` escalates on the similarity floor (ADR-009), and an operator types "no realizamos ese estudio" by hand — the same sentence, several times a day, for a fact that will not change until the catalog does. The lab asked for it to stop.

The obvious shape — a tenant-authored list of services *not* offered — was considered first and rejected in design. A lab cannot enumerate the set of studies it does not perform; that set is unbounded and arbitrarily specific, and every gap in it is a question that still reaches a person. The list a tenant *can* maintain is the one it already maintains: its own catalog.

So the catalog is treated as complete. Absence from it means the tenant does not offer the thing. That is a closed-world assumption, and it is a strong one: it converts every retrieval failure into a confident denial delivered to a patient. The rest of this ADR is the guards that make it survivable.

## Decision: deny only when two independent signals agree

**Closed world is opt-in per tenant.** `tenants.catalog_is_closed`, default false. A tenant whose corpus is prose rather than an item table would otherwise deny everything, and a tenant nobody has finished configuring is exactly the one that must not be answering on the tenant's behalf. The default costs nothing: an un-flagged tenant behaves exactly as it does today.

**Two signals, and disagreement escalates.** A denial requires both a lexical membership miss over the tenant's own item set — `document_chunks.metadata_` already carries per-item `id`/`type`/`category`/`keywords`, and `content_tsv` is an indexed Spanish `tsvector`, so this is a query, not a new store — and a maximum similarity below `handoff_threshold`. When the two disagree, the thread escalates exactly as it does now. Each signal alone is a known source of wrong denials: lexical misses every synonym, and the similarity floor is the same blunt number that ADR-009 deliberately declined to trust on its own.

**Alternative considered and rejected:** a second, higher similarity threshold meaning "definitely absent", with no lexical signal. Rejected because it answers a complaint about non-determinism with more of the same number, and because a threshold cannot distinguish "the tenant does not do this" from "the tenant was never indexed" — a distinction `generate()` already makes deliberately for the empty-pool case.

**The denial is gated on query expansion having actually run.** `retrieve.py`'s `_rewrite_query` already expands the user's words with formal synonyms drawn from the tenant's own domain text, which is what carries "biopsia de riñón" to "nefrectomía". The membership test runs against the *expanded* query. But expansion is best-effort: it times out at 10s and falls back to the raw query, and today that fallback is indistinguishable from a successful expansion with nothing to add. `_rewrite_query` returns an explicit flag so the two can be told apart. An exception or timeout blocks denial and escalates; an expansion that ran and found nothing to add does not — the model looked.

Expansion is fed `specialization_context`, falling back to `expertise_area`, falling back to a generic string. Only the first two count as expansion-grade for the gate. Both are tenant-authored domain text; the generic constant contributes no synonyms while making the flag read true, which would reopen this hole invisibly.

**The verdict is computed in `retrieve()`, not `generate()`.** That node already holds the expanded query and the chunks, and it carries a `CachePolicy(ttl=90)` keyed `tenant::question` (`builder.py:55`) — computing the verdict there caches it with the rest of the node's output. Deriving it in `generate()` instead would re-derive on every cache hit with no expanded query in hand.

**The reply is fixed text, not a generated one.** `tenants.not_offered_message`, nullable, falling back to a vertical-neutral constant. There is nothing to reason about in "we don't do that", and the whole feature exists to remove model calls from this path.

**Every denial is logged and audited** as an outcome distinct from an escalation. Automating the denial removes the human who would have noticed that the catalog says "nefrectomía" while patients say "biopsia de riñón". The log is what replaces that person; without it the failure is silent until a patient complains to the tenant.

**The OCR path denies only on a confident extraction.** `vision.py` already separates a confident reading from uncertainty. An unreadable photo continues to ask the user to type — turning "we could not read this" into "we do not offer this" is a wrong answer wearing a decision's clothes.

## Consequences

**Positive:**
- The recurring question stops reaching an operator, without a per-message model call: one SQL membership query on a path that already queries.
- The two-signal rule means the automation gives up in precisely the cases a human is useful — ambiguity, not absence.
- Denials are inspectable, so synonym gaps surface as a list to fix rather than as a complaint.

**Negative:**
- A synonym the catalog and the expansion both miss produces a confident wrong denial to a patient. This is the accepted cost of the closed-world assumption; the log bounds how long it goes unnoticed, it does not prevent it.
- The feature is inert for a tenant with neither `specialization_context` nor `expertise_area` filled. Deliberate — see the gate above — but it means "denials never fire" has a configuration explanation before it has a bug explanation.
- Three guards (two signals, the expansion gate, the log) exist to make one assumption safe. Removing any one of them silently restores the failure mode; none is incidental.
