# ADR-008: One specialized paid model per task, not one dual-purpose model

**Status:** Accepted
**Date:** 2026-08

## Context

The upgraded config (`docs/model-upgrade-baseline.md`, issue #21) had `xiaomi/mimo-v2.5` serving both chat (triage + generate) and vision — one multimodal model doing everything paid. Multimodal models are frequently priced above the best single-purpose model for either half of their capability, so a dual-purpose model is rarely the cheapest way to cover two different jobs at acceptable quality. Triage's classification call and generate's Spanish-language RAG synthesis are also different shaped tasks — one benefits from cheap structured output, the other from conversational fluency — and were paying for whichever one model's strengths didn't match the task.

Separately, the configured text fallback (`deepseek/deepseek-v4-flash`, no `:free` suffix) was actually a paid model, not the free safety net it was meant to be — the `:free` variant of that model was discontinued from OpenRouter's catalog sometime before this decision.

## Decision

Split by task, each pinned to a specialized paid model chosen for price/quality on that specific job:

- **Triage** (`triage.py`, cheap structured classification): ~~`amazon/nova-micro-v1`~~ → **`openai/gpt-4.1-nano` (see Update below)**, via new `get_triage_llm()` (`app/services/llm.py`). Originally `openai/gpt-5-nano`; swapped to `nova-micro-v1` after a real-code re-validation (`docs/model-upgrade-baseline.md`) found it beat gpt-5-nano on accuracy (10/10 vs. 9/10 on a 10-case real `triage()` run — gpt-5-nano misclassified an off-topic message as `rag`), latency (0.53s vs. 2.42s avg), and price ($0.035/$0.14 vs. $0.05/$0.40 per 1M) simultaneously — no trade-off to weigh at the time. No model-level fallback — triage already degrades safely (JSON-parse retry, then defaults to `"rag"`) without needing a second model.
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

## Update (2026-08): triage swapped from `nova-micro-v1` to `gpt-4.1-nano`

Live traffic and a real-code re-benchmark (`scripts/benchmark_model_upgrade.py`) found `amazon/nova-micro-v1` badly underperforming its original numbers above — 0.55-1.00s per call one run, 7-40s+ (occasionally worse) on live traffic, plus intermittent `triage_structured_failed` warnings. Root-caused with evidence, not assumed:

- OpenRouter's own API (`GET /api/v1/models/amazon/nova-micro-v1/endpoints`) lists this model/endpoint's `supported_parameters` as `["max_tokens", "temperature", "top_p", "top_k", "stop", "tools"]` — **`tool_choice` is not supported**. Every forced-tool `with_structured_output()` call silently loses that forcing in transit to Bedrock.
- Per AWS's own Nova docs, Nova models default to chain-of-thought (`<thinking>...</thinking>`) reasoning for tool calls, and forcing `tool_choice` is what's documented to suppress it. Without that forcing reaching the model, it intermittently "thinks" in plain text and skips the tool call entirely — reproduced directly (`with_structured_output(..., include_raw=True)` showed `tool_calls: []` with a `<thinking>` preamble in `content`).
- Separately, one of the two OpenRouter/Bedrock endpoints behind this model measured 7.8% uptime over the trailing 30 minutes (`status: -5` in the same endpoints response) — a likely contributor to the worst-case latency spikes, independent of the tool_choice gap.

Neither is fixable from our side (OpenRouter's parameter support, Bedrock's own endpoint health) — this isn't a code bug, it's the provider not supporting what the task needs.

Re-benchmarked real alternatives (same script, same 8-case set) that DO list `tool_choice` in their `supported_parameters` and don't mandate reasoning:

| Model | Accuracy | Avg latency | Notes |
|---|---|---|---|
| `amazon/nova-micro-v1` (previous) | ~75%, flaky | 0.77s avg, spikes to 40s+ | No `tool_choice` support |
| `openai/gpt-4.1-nano` | **8/8** | **0.81s** (max 1.17s) | Chosen |
| `google/gemini-2.5-flash-lite` | 8/8 | 75.83s avg — one call hung 601s past the configured 60s timeout | Rejected: unacceptable tail latency |

`triage_model` is now `openai/gpt-4.1-nano` ($0.05/$0.20 per 1M — slightly above nova-micro's nominal price, but zero observed fallback-retry calls closes most of that gap in practice). Updated in `app/config.py`, `.env`, `.env.example`.
