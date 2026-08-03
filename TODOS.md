# TODOS

Items accepted for future work but out of current PR scope.

- **WhatsApp: `WhatsAppAdapter.normalize()` is unused by the live webhook path.**
  `whatsapp_webhook`/`_handle_message` (`app/channels/whatsapp.py`) parse the raw
  payload directly instead of going through the `ChannelAdapter` Protocol, unlike
  Telegram's text path. Already flagged in code as `ponytail: ... migrate when
  adding a 3rd channel` — deferred again rather than refactored under time
  pressure while getting WhatsApp ready for testing. Revisit when adding a 3rd
  channel or when unifying the two handlers' message-type branching.
- **Feedback de usuario 👍/👎 sobre respuestas del bot.** (movido desde plan
  voz/escalabilidad, eng-review 2026-07-06 D9 — no seleccionada en decisión CEO)
  Qué: botones inline (Telegram) / reacciones (WhatsApp) → tabla feedback.
  Por qué: señal directa de calidad de respuestas para mejorar RAG/prompts.
  Pros: datos de eval reales por tenant. Cons: UI por canal x2, tabla nueva,
  volumen bajo la hace poco útil al inicio. Contexto: implementar DESPUÉS de
  3.0 media_pipeline (evita duplicar handling por canal) y de 3.4 métricas
  (comparte patrón de agregación). Empezar por: callback_query handler en
  telegram.py + tabla `response_feedback`.
- **WhatsApp: `wa_service_windows` table has no reader.** It's updated on every
  inbound message but nothing enforces Meta's 24h free-form-reply window. Not a
  live bug today — every current send is a same-turn reply to an inbound message,
  so the window is always fresh — but it will matter the moment any
  business-initiated/proactive send path exists (e.g. an operator dashboard
  replying later, or a follow-up message). Add the read-side check (and a
  template-message fallback) when that feature is built.
- **Fold `tenant.specialization_context` into triage.py's classification prompt.**
  (plan-eng-review 2026-07-24, feature/especilizacion-bot — deferred at Step 0
  complexity check: 9 core files tripped the 8-file smell, triage.py was the
  self-flagged weakest cut candidate)
  Qué: pass the tenant's specialization_context (already loaded from
  `state["tenant_id"]`, same pattern `generate.py:105` uses) into
  `_TRIAGE_PROMPT` (`app/graph/nodes/triage.py`) so jargon-heavy messages
  classify correctly from the first step, not just in the final response.
  Por qué: `_TRIAGE_PROMPT` already defaults ambiguous cases to `"rag"`
  (`triage.py:29`), so the misclassification risk without this is low —
  that's exactly why it was deferred rather than cut outright. Pros: cheap
  once needed (no new plumbing — `AgentState` already carries `tenant_id`).
  Cons: one more DB round trip on the message-classification critical path.
  Depends on: nothing — `specialization_context` column and the injection
  pattern in `generate.py`/`vision.py` already shipped. Resume when there's
  evidence of real triage misclassification on jargon-heavy messages.
- **Admin "preview/test prompt" panel.** (plan-ceo-review 2026-07-24,
  feature/especilizacion-bot — cherry-pick D3.2, deferred)
  Qué: a panel in admin.html where an operator types a sample question or
  uploads a sample image and sees how the bot would respond with the
  current `specialization_context`, before saving — without touching live
  traffic. Por qué: the only way to verify a specialization_context edit
  actually improved anything today is a real chat/photo test in production.
  Pros: directly closes the "no sé si funcionó" gap for operators; useful
  for any tenant, not just the one that prompted this feature. Cons: new
  surface (endpoint + UI + test-image upload handling), doubles LLM/vision
  cost per preview. Depends on: the base `specialization_context` field
  (shipped 2026-07-24) being live long enough to know how often operators
  actually need to iterate on the text.
- **Webhook endpoints have no rate limiting.** (/qa 2026-08-03) `slowapi`'s
  `Limiter` is only applied to `/operator/resume` (`20/minute`). An
  unauthenticated POST flood against a guessed/leaked tenant slug on
  `/webhook/telegram/{slug}` or `/webhook/whatsapp/{slug}` isn't throttled
  at the app layer — auth there is a per-tenant secret, not IP-based, so it
  may be an intentional gap, but it wasn't a deliberate decision anyone
  made. Revisit if webhook abuse shows up in logs/cost.
- **`retrieve_chunks()` and `_call_rerank_api()` aren't wrapped in OTel
  spans.** (/qa 2026-08-03, Phoenix trace review) They're plain
  httpx/SQLAlchemy calls, not LangChain Runnables, so `auto_instrument=True`
  never captures them — a 37.7s trace during this QA session showed ~3s of
  visible ChatOpenAI spans and 30+s of unattributed dead time between
  `triage` and `generate`, almost certainly the embedding/rerank call
  against a free-tier OpenRouter model. Add explicit spans around both so
  the next latency spike is diagnosable instead of invisible. See
  `.gstack/qa-reports/qa-report-langraph-bot-v1-2026-08-03.md`.
