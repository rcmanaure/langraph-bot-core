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

`triage_model` was set to `openai/gpt-4.1-nano` ($0.10/$0.40 per 1M — the real-time endpoint, not the `:batch` variant, which is async/24h-turnaround and unusable here) — superseded by the next update below within the same day.

## Update (2026-08): triage swapped again, `gpt-4.1-nano` → `meta-llama/llama-3.1-8b-instruct`

A follow-up review (3 parallel Haiku subagents, one per model category, each independently researching OpenRouter's catalog and real-code-benchmarking candidates) found `meta-llama/llama-3.1-8b-instruct` beating `gpt-4.1-nano` on every axis for triage. Verified independently (not taken on the subagent's word alone) with 24 real calls across 2 runs, 12 cases each:

| Model | Accuracy | Avg latency | Max latency | Price (prompt/completion per 1M) |
|---|---|---|---|---|
| `openai/gpt-4.1-nano` (previous) | 8/8 | 0.81s | 1.17s | $0.10 / $0.40 |
| `meta-llama/llama-3.1-8b-instruct` | **24/24** | **0.30-0.44s** | 0.96s | **$0.05 / $0.08** |

Half the price, roughly 2x faster, no accuracy loss, confirmed `tool_choice` support via OpenRouter's models API. `triage_model` is now `meta-llama/llama-3.1-8b-instruct`. Updated in `app/config.py`, `.env`, `.env.example`.

The same review round also benchmarked `generate` (chat) and `vision` model alternatives. Both subagents independently recommended `google/gemini-2.5-flash-lite` (cheaper/faster than the current `mistral-small-3.2-24b-instruct` and `qwen3-vl-32b-instruct` picks) — **not adopted for either.** A direct stress test (12 sequential real calls) came back clean, but an earlier triage-benchmark run against this same model hit one call that took 601 seconds — 10x past the configured 60-second client timeout — and returned normally (not a timeout exception), meaning our own timeout safety net didn't fire at all for that call. `generate()` runs on every single RAG turn with no independent timeout/fallback for a hang shaped like that (its existing fallback triggers on exception, not on a slow-but-successful response), so one silent 10-minute stall in production is a worse failure mode than the flakiness this whole ADR update chain has been fixing. Not ruled out permanently, but not adopted until that hang is understood or reproduced reliably enough to bound its risk.

## Update (2026-08-18): full sweep, all three categories, hard price cap

Standing review rule going forward: **any candidate must be < $0.50 per 1M tokens for both prompt and completion, real-time (non-`:free`) tier, confirmed via a live pull of `https://openrouter.ai/api/v1/models` — not a subagent's estimate.** Re-ran the 3-subagent-fleet pattern (one Haiku agent per category) under this explicit cap, then independently re-verified every price and every latency claim myself before deciding whether to apply anything (a subagent had already been caught quoting invented/estimated prices in this same round — see vision finding below).

**Triage — no change.** Candidates `nvidia/nemotron-3.5-lightning` ($0.08/$0.20), `inclusionai/ling-3.0-flash` ($0.021/$0.063), `upstage/solar-pro4` ($0.03/$0.12) — all three prices spot-checked against the live OpenRouter catalog and confirmed exact. All confirmed `tool_choice` support, all real-benchmarked (22 calls each, 11 cases × 2 runs). None beat current `meta-llama/llama-3.1-8b-instruct` (0.29s avg, 100% accuracy) — `solar-pro4` in particular spiked to 7.8s max on one call. `triage_model` unchanged.

**Generate — no change, evidence conflict.** Candidate `mistralai/mistral-nemo` is real and ~5x cheaper than current ($0.019/$0.03 vs. current `mistral-small-3.2-24b-instruct`'s $0.094/$0.25 — both prices live-verified). Subagent claimed it 37% *faster* (1.27s vs. 2.07s avg) with clean tail latency. My own independent spot-check (6 real calls each, same 3-case set, back-to-back) found the opposite: current model averaged 1.40s, `mistral-nemo` averaged 2.33s — slower, not faster. Accuracy was 100% both ways in every run. Given the direct contradiction on the one axis that mattered for the subagent's recommendation, **not applying the switch** — small sample sizes (3-6 calls) on both sides make this noise until someone runs a real 20+ call trial on each. `openai_model` unchanged. If revisited: `mistral-nemo`'s price advantage alone may justify adoption even at parity latency, given both are well under 3s for a chat turn — just needs a bigger real sample first.

**Vision — no change, subagent skipped the required real-image test.** Instructions explicitly required benchmarking candidates against the real photos in `test_images/` through `extract_procedure_query()`. The subagent instead did desk research only and reported *estimated* pricing for the current model (`~$0.25/$1.25 per 1M`) that turned out to be wrong by ~3x when checked against the live catalog (actual: `qwen/qwen3-vl-32b-instruct` is $0.104/$0.416). Three real, vision-capable, in-budget candidates were correctly identified though — `google/gemini-3.6-flash` ($0.00075/$0.00375 — well under cap, general-purpose multimodal not vision-specialized), `qwen/qwen3.7-flash` ($0.00003/$0.00013), `bytedance-seed/seed-2-1-turbo` ($0.0005/$0.0025, 262K context) — all confirmed to exist and support image input via the live API. None of these were actually run against the 11 real patient photos, so there's no accuracy evidence either way. Current model kept unchanged (safer default given this is a real-PII/PHI extraction task and the prior `qwen3-vl-8b` real-image test already showed a smaller sibling model degrades on hard cases). If revisited: these three are worth a real `test_images/` run before deciding, same protocol as the `qwen3-vl-8b` comparison earlier in this doc.

**Lesson for future rounds:** a subagent's stated price or latency number is not evidence — the price must be pulled live from `/api/v1/models` and the latency must be independently re-run, not just quoted, before it can justify a production config change. Two of three subagents in this round needed correction for exactly this reason.

**Quick-reference (current, live-verified 2026-08-18) — check this table first; only re-run the full fleet review if a model listed here gets delisted, repriced above $0.50/1M, or a real production incident points at it:**

| Category | Config field | Current model | Prompt / Completion per 1M |
|---|---|---|---|
| Triage | `triage_model` | `meta-llama/llama-3.1-8b-instruct` | $0.05 / $0.08 |
| Generate | `openai_model` | `mistralai/mistral-small-3.2-24b-instruct` | $0.094 / $0.25 |
| Vision | `openai_vision_model` | `qwen/qwen3-vl-32b-instruct` | $0.104 / $0.416 |
