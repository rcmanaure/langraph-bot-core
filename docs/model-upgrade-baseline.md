# Model upgrade baseline (issue #21)

Measured with `scripts/benchmark_model_upgrade.py`: 3 known-answer chat
questions against a small fixed catalog context, 1 rerank call over 3
candidate documents, 1 vision round-trip against a synthetic printed-text
image (latency only — no real labeled doctor's-order image is available in
this environment; see the script's own docstring).

## Before (free/lite tier)

| Stage  | Model                                          | Accuracy | Avg/only latency |
| ------ | ----------------------------------------------- | -------- | ----------------- |
| Chat   | `nvidia/nemotron-3-super-120b-a12b:free`        | 3/3      | 8.36s avg (4.54s / 18.85s / 1.68s) |
| Vision | `google/gemini-3.1-flash-lite-image-20260630`   | n/a      | 1.07s |
| Rerank | `nvidia/llama-nemotron-rerank-vl-1b-v2:free`    | correct  | 1.24s |

## After (upgraded)

| Stage  | Model                  | Accuracy | Avg/only latency |
| ------ | ----------------------- | -------- | ----------------- |
| Chat   | `xiaomi/mimo-v2.5`      | 3/3      | 13.77s avg (4.21s / 22.79s / 14.3s) |
| Vision | `xiaomi/mimo-v2.5`      | n/a      | 12.09s |
| Rerank | `cohere/rerank-v3.5`    | correct  | 0.92s |

## Findings

- **Chat/vision got slower, not faster**, in this small (n=3) sample —
  13.77s vs 8.36s avg for chat, 12.09s vs 1.07s for vision. The upgrade's
  real value here is getting off a free/lite tier with its own rate limits
  and no uptime guarantee, not a latency win — this run doesn't support
  claiming one. Worth a larger sample before treating either number as
  representative; free-tier providers can also be unusually fast when
  under-utilized, which the baseline run may have caught.
- **Rerank improved** — 0.92s vs 1.24s, and correctness held.
- `xiaomi/mimo-v2-flash` (the originally chosen cheaper chat model) is
  deprecated on OpenRouter as of this run; its API response recommends
  `xiaomi/mimo-v2.5`, which is what's configured for chat here (same model
  now serves both chat and vision).
- Accuracy checks are narrow (3 fixed catalog questions, 1 rerank query) —
  they catch a model that stops following instructions entirely, not subtle
  quality regressions. Vision has no accuracy check at all in this
  environment, only latency.
