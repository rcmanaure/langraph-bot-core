# ADR-008: One specialized paid model per task, not one dual-purpose model

**Status:** Accepted
**Date:** 2026-08

## Context

The upgraded config (`docs/model-upgrade-baseline.md`, issue #21) had `xiaomi/mimo-v2.5` serving both chat (triage + generate) and vision — one multimodal model doing everything paid. Multimodal models are frequently priced above the best single-purpose model for either half of their capability, so a dual-purpose model is rarely the cheapest way to cover two different jobs at acceptable quality. Triage's classification call and generate's Spanish-language RAG synthesis are also different shaped tasks — one benefits from cheap structured output, the other from conversational fluency — and were paying for whichever one model's strengths didn't match the task.

Separately, the configured text fallback (`deepseek/deepseek-v4-flash`, no `:free` suffix) was actually a paid model, not the free safety net it was meant to be — the `:free` variant of that model was discontinued from OpenRouter's catalog sometime before this decision.

## Decision

Split by task, each pinned to a specialized paid model chosen for price/quality on that specific job:

- **Triage** (`triage.py`, cheap structured classification): `amazon/nova-micro-v1`, via new `get_triage_llm()` (`app/services/llm.py`). Originally `openai/gpt-5-nano`; swapped after a real-code re-validation (`docs/model-upgrade-baseline.md`) found `nova-micro-v1` beat it on accuracy (10/10 vs. 9/10 on a 10-case real `triage()` run — gpt-5-nano misclassified an off-topic message as `rag`), latency (0.53s vs. 2.42s avg), and price ($0.035/$0.14 vs. $0.05/$0.40 per 1M) simultaneously — no trade-off to weigh. No model-level fallback — triage already degrades safely (JSON-parse retry, then defaults to `"rag"`) without needing a second model.
- **Generate** (`generate.py`, Spanish RAG synthesis): `mistralai/mistral-small-3.2-24b-instruct`, via `get_chat_llm()` (renamed in meaning, not signature). Fallback on failure: `nvidia/nemotron-3-super-120b-a12b:free` — a real `:free` model, and the same one already proven 3/3 on `benchmark_model_upgrade.py`'s accuracy check as the pre-upgrade baseline.
- **Vision** (`vision.py`): a vision-specialized model (`qwen/qwen3-vl-32b-instruct` recommended — see `docs/model-upgrade-baseline.md`), no longer shared with chat. No fallback — an image that can't be read resolves to uncertainty and the user is asked to type instead (see `Extraction` in `CONTEXT.md`), which is the existing behavior, not a new one.
- **Embedding and rerank** were already single-purpose paid models (`openai/text-embedding-3-small`, `cohere/rerank-v3.5`) and are unchanged by this decision.
- **Greeting** already never reaches a model at all — `triage.py`'s regex shortcut and `generate.py`'s static `_GREETING_MSG` predate this ADR and aren't touched by it.

`retrieve.py`'s query-expansion rewrite and `update_profile.py`'s extraction call are structurally similar to triage (cheap, structured) but stayed on the general `get_chat_llm()` model — splitting them out wasn't part of this decision and can be revisited separately if their cost becomes worth optimizing.

**Alternative considered and rejected:** keep one multimodal model for chat+vision, just swap it for a cheaper multimodal alternative. Rejected because the price/quality-per-task principle this ADR exists to establish is specifically that a job-matched single-purpose model beats a multimodal generalist on cost, quality, or both — and picking a cheaper generalist doesn't test that.

## Consequences

**Positive:**
- Each task is billed and evaluated against a model actually suited to it, rather than inheriting whatever the chat model happened to be.
- The text fallback is now genuinely free, closing a real billing gap.

**Negative:**
- Four model IDs to track instead of one (plus a fallback) — `docs/model-upgrade-baseline.md` is the source of truth for what's currently configured and why.
- OpenRouter's free-tier catalog churns (models delisted without notice — this ADR exists partly because that already happened once). The `nvidia/nemotron-3-super-120b-a12b:free` fallback should be re-verified periodically, not treated as permanent.
