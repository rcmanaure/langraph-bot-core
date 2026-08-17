# Model upgrade baseline (issue #21, ADR-008)

Measured with `scripts/benchmark_model_upgrade.py`: 4 triage classification
calls, 3 known-answer chat questions against a small fixed catalog context,
1 rerank call over 3 candidate documents, 1 vision round-trip against a
synthetic printed-text image (latency only — no real labeled doctor's-order
image is available in this environment; see the script's own docstring).

## Before (free/lite tier — pre-issue-#21)

| Stage  | Model                                          | Accuracy | Avg/only latency |
| ------ | ----------------------------------------------- | -------- | ----------------- |
| Chat   | `nvidia/nemotron-3-super-120b-a12b:free`        | 3/3      | 8.36s avg (4.54s / 18.85s / 1.68s) |
| Vision | `google/gemini-3.1-flash-lite-image-20260630`   | n/a      | 1.07s |
| Rerank | `nvidia/llama-nemotron-rerank-vl-1b-v2:free`    | correct  | 1.24s |

## Intermediate (dual-purpose paid model — superseded by ADR-008)

| Stage  | Model                  | Accuracy | Avg/only latency |
| ------ | ----------------------- | -------- | ----------------- |
| Chat   | `xiaomi/mimo-v2.5`      | 3/3      | 13.77s avg (4.21s / 22.79s / 14.3s) |
| Vision | `xiaomi/mimo-v2.5`      | n/a      | 12.09s |
| Rerank | `cohere/rerank-v3.5`    | correct  | 0.92s |

One model (`xiaomi/mimo-v2.5`) served both chat and vision here — the setup
ADR-008 replaced. See ADR-008 for why: multimodal models are frequently
priced above the best single-purpose model for either half of what they do,
and triage vs. generate are differently-shaped tasks (cheap classification
vs. Spanish fluency) that were both paying for one model's compromise.

## Current (specialized per-task paid models — ADR-008)

| Stage    | Model                                        | Fallback | Accuracy | Avg/only latency | Notes |
| -------- | --------------------------------------------- | -------- | -------- | ----------------- | ----- |
| Triage   | `openai/gpt-5-nano`                          | none — degrades to `"rag"` on failure, no model swap | 4/4 | 2.18s avg | Cheap, strong structured output |
| Generate | `mistralai/mistral-small-3.2-24b-instruct`   | `nvidia/nemotron-3-super-120b-a12b:free` | 3/3 | 0.94s avg | Chosen for Spanish fluency/tone over raw reasoning; also fastest chat model measured across all three runs in this doc |
| Vision   | `qwen/qwen3-vl-32b-instruct`                 | none — illegible image resolves to uncertainty (see `Extraction` in `CONTEXT.md`) | n/a (latency only) | 1.22s | OCR-strong, ~10x cheaper than the most accurate handwriting option (`anthropic/claude-haiku-4.5`); escalate only if real-world illegibility failures show up |
| Embed    | `openai/text-embedding-3-small`              | n/a      | —        | —                  | Unchanged — already single-purpose |
| Rerank   | `cohere/rerank-v3.5`                         | n/a      | correct  | 1.25s              | Unchanged — already single-purpose |
| Greeting | none — regex shortcut in `triage.py` + static reply in `generate.py`, no model call at all | n/a | n/a | n/a | Predates ADR-008 |

Measured 2026-08-17 via `.venv/Scripts/python.exe scripts/benchmark_model_upgrade.py`.
First triage run scored 2/4 — the benchmark script's own `_bench_triage()`
was sending the bare query with no system prompt, so the model was guessing
the category blind. Fixed to send the same `_TRIAGE_PROMPT` triage.py
actually uses; re-run scored 4/4. The bug was in the benchmark, not the
model — worth remembering before reading a low first-run number as a model
quality signal.

The vision sample response's "Ñ" printed as a replacement character
(`PULM�N`) in the terminal — a console encoding artifact of printing the
response, not a signal the model misread the image.

## Findings (superseded runs kept for history)

- **Chat/vision got slower, not faster** going from free-tier to the
  intermediate dual-purpose paid model, in this small (n=3) sample —
  13.77s vs 8.36s avg for chat, 12.09s vs 1.07s for vision. That upgrade's
  real value was getting off a free/lite tier's rate limits and lack of
  uptime guarantee, not a latency win.
- **Rerank improved** in the same move — 0.92s vs 1.24s, correctness held.
- The originally-configured text fallback (`deepseek/deepseek-v4-flash`,
  no `:free` suffix) turned out to be a paid model — its `:free` variant
  had been discontinued from OpenRouter's catalog. ADR-008's fallback
  (`nvidia/nemotron-3-super-120b-a12b:free`) was verified live against
  OpenRouter's free-models collection and has known accuracy history in
  this repo (it's the pre-upgrade baseline chat model above).
- Accuracy checks remain narrow (4 triage cases, 3 catalog questions, 1
  rerank query) — they catch a model that stops following instructions
  entirely, not subtle quality regressions. Vision has no accuracy check
  at all in this environment, only latency.
- OpenRouter's free-tier catalog churns (delistings without notice) — the
  free fallback should be re-verified periodically, not treated as
  permanent (see ADR-008's Consequences).
