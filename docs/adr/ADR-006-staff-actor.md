# ADR-006: The Staff Member as a First-Class Actor

**Status:** Accepted
**Date:** 2026-08

## Context

Every message the bot receives so far has come from the same kind of actor: a person making an enquiry. Upcoming work (staff-only order-status lookup) needs a second kind — a staff member — with different privileges. Before that lookup can exist, the system needs a way to decide, for a given message, whether the sender is staff.

## Decision: Identity comes from the channel, never from the conversation

A staff member is a row an operator creates in the admin panel: a tenant, a channel, and that channel's identifier for the person (a Telegram user id, a WhatsApp phone number). The inbound turn resolves membership once per turn — after the message is parsed, before the graph runs — by looking up `(tenant_slug, channel, user_id)` against that allowlist. The result travels into `AgentState` as `is_staff` and is otherwise inert: nothing about it can be set, extended, or overridden by anything in the message.

**Alternative considered and rejected:** let a person declare staff status in the message itself ("soy del staff", a shared code word) and have an LLM or a keyword match recognize it. Rejected because a claim in prose is exactly what an unauthorized sender would also produce — the channel gives no way to distinguish a true claim from a false one, so any prose-based check is a privilege escalation waiting to be used. The channel-supplied identifier (who actually holds the Telegram account or WhatsApp number) is the only signal in a webhook payload that cannot be forged by writing different words.

An empty allowlist resolves every identity to non-staff, so a tenant that has never used this feature sees no behavior change — staff-only features simply have no one to grant access to yet.

## Consequences

**Positive:**
- Staff-gated behavior (like the order-status lookup) can trust `state["is_staff"]` without re-deriving or re-checking identity itself — resolution happens exactly once, at the turn.
- The allowlist is scoped per tenant and per channel by construction (the lookup key includes both), so a staff grant on one tenant's Telegram bot cannot leak to another tenant, or from Telegram to WhatsApp for the same person.
- No new attack surface in the LLM prompt path: nothing about staff status is ever fed to a model or extracted from generated text.

**Negative:**
- Onboarding a staff member is a manual admin-panel step per channel — a staff member active on both Telegram and WhatsApp needs two allowlist rows, one per identifier. This is deliberate (channel identifiers aren't fungible) but is more setup than a single shared secret would be.
- A DB lookup is added to every turn that reaches the graph. Resolution fails closed (a DB error resolves to non-staff, logged, never raised) so an outage degrades to "no one is staff" rather than blocking replies or granting access it can't verify.
