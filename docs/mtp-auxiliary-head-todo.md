# MTP Auxiliary Head TODO

DeepSeek-V3 reports Multi-Token Prediction (MTP) as a training objective and as
part of speculative decoding: https://arxiv.org/abs/2412.19437. For
`limes-nanogpt`, the clean first step is an auxiliary training loss only.

Do not implement speculative decoding, MTP KV-cache reuse, or GLM-style
IndexShare behavior until there is a primary source and a local benchmark. The
GLM-5.2/IndexShare KV-reuse note is currently treated as unverified.

## Proposed Design

- Add an optional config flag such as `mtp_depth = 0` by default.
- When `mtp_depth > 0`, keep the normal next-token loss unchanged.
- Add one or more auxiliary linear heads over the final hidden state for tokens
  at offsets `+2`, `+3`, and so on.
- Mask positions that do not have a future target.
- Save `mtp_depth` and `mtp_loss_weight` in `run_config.json` and checkpoints.
- Keep generation unchanged until a separate speculative decoding experiment is
  designed and benchmarked.

## Acceptance Criteria

- `python3 -m unittest` includes a test that `mtp_depth=0` preserves the
  existing `GPT.forward(idx, targets)` API and loss behavior.
- A gated test builds a tiny `GPTConfig` with `mtp_depth=1`, runs one forward
  pass on CPU, and verifies the auxiliary loss contributes only when requested.
- `config/train_mtp_smoke.py` runs on CPU in roughly the same class as
  `config/train_smoke.py`.
- The docs and experiment log state that MTP is an auxiliary-loss exploration,
  not an inference speedup or quality claim.
- Any future KV-cache reuse experiment cites a primary source and compares
  generated tokens against recomputation on the same prompts.
