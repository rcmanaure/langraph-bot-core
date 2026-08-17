# ADR-007: The Register Floor as a System-Owned Constraint

**Status:** Accepted
**Date:** 2026-08

## Context

The shipped default tone was "cálido y cercano... emojis casuales están bien" — a friendly-chat register, applied uniformly to patients and lab staff discussing medical tests. Health contexts don't want that, and the same free-text `tone_description` field a tenant operator edits in the admin panel was the *only* thing standing between "professional" and "chatty" — an operator could type anything and the bot's register would follow, including back into informal Spanish or emoji.

Three further defects lived in the same prompts: a name hint that greeted people using a stale stored name (fixed separately, see the deleted profile-name enrichment), a catalog prompt that simultaneously demanded "every item" and "max 4-5 lines," and per-chunk match-confidence labels (`[COINCIDENCIA EXACTA]` / `[APROXIMACIÓN...]`) spliced into the LLM's context with instructions not to leak them — bookkeeping the code already had, handed to the model to narrate and then told not to repeat.

The prompts also mixed `vos` and `tú` conjugations in the same file, so even a correct answer read as written by two different people.

## Decision: A non-overridable floor, injected ahead of tenant tone

`_REGISTER_FLOOR` (`app/graph/nodes/generate.py`) is fixed template text — not tenant data — prepended to `_FORMAT_HINT`, which both `_RAG_SYSTEM` and `_CATALOG_SYSTEM` always include: `usted` throughout, no emoji, no diminutives, no filler opener, never address by first name. The tenant's `tone_description` is interpolated as a bounded phrase *inside* this fixed structure ("Tono: {tone_description}") — it can colour delivery but cannot remove or contradict a rule that isn't stored as tenant data in the first place. `DEFAULT_TONE_DESCRIPTION` itself was rewritten to a formal register so a tenant that never touches the field gets the floor's voice by default, not the old chatty one.

Every prompt in the system that addresses the user or instructs the model in Spanish (`generate.py`, `vision.py`'s extraction/verification prompts, the static strings in `app/channels/turn.py`) was normalized to consistent `usted` conjugations. `triage.py` and `update_profile.py` stayed as-is — their prompts are English instructions to the model with no Spanish forms of address.

**Alternative considered and rejected:** keep the floor as another instruction inside the tenant's editable tone text (a "please stay formal" suffix). Rejected because that's exactly the shape of the bug being fixed — an instruction living in the same free-text channel a tenant operator controls is one edit away from being contradicted.

## Consequences

**Positive:**
- No admin-panel edit to `tone_description` can produce an informal or emoji-laden reply — the floor isn't reachable from tenant-controlled data.
- The register is expressed once, in one constant, rather than requiring every prompt to separately remember to be formal.

**Negative:**
- The prompts remain lab-specific (clinical vocabulary in `_TRIAGE_PROMPT`, extraction examples in `vision.py`). Generalizing to a second, differently-worded tenant vertical is deferred — see the `specialization_context`-in-triage item in `TODOS.md`, which is the intended migration path when that's needed.
