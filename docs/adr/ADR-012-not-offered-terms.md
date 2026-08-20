# ADR-012: A Deterministic Not-Offered Term List, in Front of ADR-010

**Status:** Accepted
**Date:** 2026-08

## Context

ADR-010 considered and rejected a tenant-authored list of services *not* offered, as the primary mechanism for a closed-world denial: "A lab cannot enumerate the set of studies it does not perform; that set is unbounded and arbitrarily specific, and every gap in it is a question that still reaches a person." It chose instead a two-signal rule (lexical catalog miss + embedding similarity floor, gated on query expansion) that reasons from what the tenant *does* offer.

In production, that mechanism turned out to almost never fire for this tenant: its embedding space clusters same-vertical medical vocabulary closely enough that `max_similarity` stays above `handoff_threshold` for genuinely-absent studies too, so the lexical/similarity signals rarely agree. Separately (see ADR-008's 2026-08-20 update), the query-expansion call the gate depends on was found hanging indefinitely in production, meaning the gate had never actually opened at all until that was fixed. Even after both are working, the operator does not want *any* model or embedding call paying for a question about a term they already know, with certainty, is not offered — not because ADR-010's mechanism is unreliable, but because a known answer shouldn't cost a model call at all.

This is the same list shape ADR-010 rejected. Reintroducing it without explaining why is exactly the failure mode `docs/agents/domain.md` warns about: a future reader of ADR-010 would reasonably conclude this design was already tried and rejected, while it ships anyway.

## Decision: the list is a fast path in front of ADR-010, not a replacement for it

**The objection ADR-010 raised was about *completeness*, not about the existence of a list.** ADR-010's rejection reasoning is entirely about what happens when the list is treated as the sole source of truth: every term missing from an unbounded, arbitrarily-specific list becomes a silent false negative — a real absence the system fails to catch. That risk is real for a list used as the *only* signal. It does not apply to a list used as a *narrower, additional, opt-in* fast path that is allowed to be incomplete, because everything the list misses still falls through to exactly the same handling as before this ADR: ADR-010's two-signal mechanism where configured, or the normal LLM/RAG turn otherwise. The list never has to be complete for the system as a whole to stay correct — it only has to be non-wrong for the terms an operator chose to put in it.

**Checked before any model call, in `triage.py`, same position `canned_answers` already occupies.** Not folded into `retrieve()`'s ADR-010 gate — that gate's entire cost (expansion call, embedding search, rerank) is exactly what this feature exists to skip for a term the operator has already, explicitly, and confidently declared absent.

**Operator-curated, not tenant-inferred.** ADR-010's "unbounded and arbitrarily specific" concern was about a system trying to derive or maintain completeness automatically. This list makes no such claim: it is exactly as complete as the operator has bothered to make it, uploaded via Excel or edited one term at a time, and an empty list is simply inert (same shape `canned_answers` already has — no separate enable/disable toggle).

**Staff are exempted, mirroring ADR-010's own `is_staff` exemption in `generate.py`.** A staff member's own message asking about a not-offered term (e.g. checking the catalog) must reach the normal path, not get auto-denied like a customer.

**Every match is still audited**, into the same `not_offered_denials` table ADR-010's mechanism writes to, with `max_similarity = NULL` distinguishing "matched a configured term, nothing was computed" from ADR-010's "computed a similarity score and it missed." ADR-010's own stated rationale for this table — "the mechanism for discovering synonym gaps" — applies identically here: a wrong entry in the list is exactly the kind of mistake this log exists to surface.

**Alternative considered and rejected: fold this into `catalog_is_closed`.** Rejected because tying a new, independent, deterministic mechanism to a toggle that means something else (whether ADR-010's LLM/embedding gate is active) is a footgun waiting to happen — an operator who never enables `catalog_is_closed` would have a fully-configured term list silently do nothing, with no error to notice.

## Consequences

**Positive:**
- Zero model/embedding/RAG cost for exactly the terms an operator already knows, with certainty, are absent — closing the actual complaint (token cost, reply latency, wording inconsistency) without needing ADR-010's mechanism to be reliable first.
- ADR-010's mechanism is untouched and still catches everything the list doesn't cover — the completeness risk that ADR-010 correctly identified stays exactly as bounded as it already was.
- The list is genuinely incremental: an operator can start with zero terms (no behavior change) and add exactly the recurring questions they're tired of answering by hand.

**Negative:**
- Two independent not-offered mechanisms now exist in the codebase (this list, and ADR-010's two-signal rule) with different trigger conditions and different failure modes. A future reader debugging "why did this customer get denied" must check both.
- The list can be wrong the same way any operator-entered data can be wrong — a keyword that's too broad denies something the tenant actually does offer, with no ambiguity check the way ADR-010's two-signal rule has one. This cost is accepted deliberately: it is the same trade-off `canned_answers` already made for its own keyword matching, and the audit log is the same mitigation ADR-010 already relies on for its own equivalent risk.
