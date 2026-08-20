# ADR-011: Prompt Packs Keyed by Vertical

**Status:** Accepted
**Date:** 2026-08

## Context

ADR-007 ended by naming what it had deferred: the prompts remain lab-specific — clinical vocabulary in `_TRIAGE_PROMPT`, extraction examples in `vision.py` — and generalizing them was left until a second, differently-worded tenant existed. That tenant is now real, though undated.

Two things in the code are worse than lab-specific prompts, and they are the reason this cannot wait for the signature. `_GREETING_MSG` (`generate.py:109`) hardcodes *this lab's* name, phone number and Google Maps URL as the fallback greeting for every tenant whose `greeting_message` is NULL. `_RESULTS_NOTE_RULE` (`generate.py:73`) appends "Resultados: 3 a 5 días hábiles" to every priced reply, for every tenant. These are not vocabulary; they are one business's facts compiled into shared code, and tenant #2 would ship quoting them on day one.

The tenant also asked, reasonably, that prompt content not keep growing inside the codebase as clients are added.

## Decision: a per-vertical pack of vocabulary, filling slots that already exist

**`tenants.vertical` selects a prompt pack.** One pack serves many tenants — two labs share `medical_lab` — so pack count is bounded by verticals, not by clients. Per-tenant prompt copies were rejected: they are exactly the growth the tenant is worried about, and a prompt bug would then need fixing once per client.

"Vertical" is not a new coinage. The codebase already reasons in it — `tenant.py:9` ("a code change per new tenant vertical"), `retrieve.py:162` ("no hardcoded per-vertical glossary"), `vision.py:514`, and ADR-007 itself. This promotes the word from comment prose to a column and a glossary term.

**Distinct from `expertise_area`.** That field is a short display label, shown in greetings, off-topic replies and the admin table (`tenant.py:26`). It names the business to a person. The vertical names the vocabulary to the system. Overloading one field to do both is how it stops doing either well.

**Only vocabulary slots become data.** Triage examples, extraction examples, item-type names. The instruction skeleton and `_REGISTER_FLOOR` stay in code.

**Alternative considered and rejected:** whole prompt templates as editable rows. Rejected because it recreates ADR-007's bug at full scale. That ADR exists because a tenant-editable free-text field was the only thing holding the register in place; an editable skeleton can delete the floor outright. A pack colours what the model recognizes and can never change how it addresses anyone.

This turns out to be cheaper than expected. `_RAG_SYSTEM` is already fully parameterized — `{expertise}`, `{tone_description}`, `{specialization_block}`, `{match_instruction}`, `{negative_confirmation_rule}`, `{format_hint}` — and `vision.py` already threads `specialization_context` through extraction and verification. Pack fields drop into seams the code already has. Only `_TRIAGE_PROMPT` is a flat constant needing new ones.

**The two hardcoded lab facts move to tenant columns, not to the pack.** `_GREETING_MSG`'s default is rewritten vertical-neutral, with this lab's text moved into its own `greeting_message` row. The results turnaround becomes a nullable `tenants` column, omitting the line when unset — turnaround varies between two labs in the same vertical, so it is tenant data, not vertical data.

**The three per-tenant free-text fields stay as they are.** `tone_description`, `specialization_context` and `expertise_area` already work and already layer on top of the shared prompts. The pack sits underneath them; it does not replace them.

## Consequences

**Positive:**
- A second tenant in a new vertical needs a pack row, not a code change — which was ADR-007's stated migration path.
- No tenant's prompt content can leak into another's replies, and no tenant's business facts sit in shared constants.
- The register floor keeps the property ADR-007 bought: it is not reachable from any data a tenant or operator can edit.

**Negative:**
- Prompt content now lives in two places — skeletons in code, vocabulary in rows — so reading a prompt end to end means reading both. The alternative was one place that a tenant edit could break.
- The packs are being designed against one real vertical and one imagined one. The seams will be wrong somewhere; the first genuinely different tenant is what proves them, and that tenant has no date.
