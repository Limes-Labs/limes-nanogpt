# RL And Efficiency Roadmap

This repo is a nanoGPT-scale research lab. It should organize small,
reproducible work without implying frontier capability.

## Sources Inspected

- Chinese AI Research Catalog: https://www.himanshustwts.com/chinese-frontier/catalog.md
- PPO: https://arxiv.org/abs/1707.06347 and https://spinningup.openai.com/en/latest/algorithms/ppo.html
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-V3 / MTP: https://arxiv.org/abs/2412.19437
- Qwen GSPO: https://arxiv.org/abs/2507.18071 and https://qwenlm.github.io/blog/gspo/
- ByteDance DAPO: https://arxiv.org/abs/2503.14476
- MiniMax CISPO: https://arxiv.org/abs/2506.13585
- Kimi k1.5 long-context RL: https://arxiv.org/abs/2501.12599
- OpenAI Parameter Golf: https://github.com/openai/parameter-golf
- AutoResearch-RL: https://arxiv.org/abs/2603.07300

## PPO Vs GRPO

PPO is a policy-gradient family that uses clipped or KL-penalized updates to
avoid moving the new policy too far from the policy that sampled the data:
https://arxiv.org/abs/1707.06347. The OpenAI Spinning Up implementation presents
PPO with both a policy and value function, using the value function to estimate
advantages and fit returns: https://spinningup.openai.com/en/latest/algorithms/ppo.html.
That value/critic path can be useful when rewards are richer, delayed, or
agentic, because a learned value estimate can reduce variance across long or
partially observed trajectories: https://spinningup.openai.com/en/latest/algorithms/ppo.html.

GRPO, introduced with DeepSeekMath, is described as a PPO variant that removes
the critic/value model and estimates a baseline from grouped samples, reducing
training resource needs for math-style verifiable rewards:
https://arxiv.org/abs/2402.03300. That makes GRPO a better conceptual fit for
small exact-match or unit-test rewards than for subjective preference rewards
that need a learned reward or value model: https://arxiv.org/abs/2402.03300.

This PR starts with `rl/ppo_toy.py` rather than a GRPO trainer because PPO's
sample, reward, clipped update, value estimate, and KL-control pieces are easier
to inspect in a tiny standalone script. A next GRPO toy should reuse the same
reward function but sample groups per prompt and replace the value estimate with
within-group normalized rewards, following the GRPO direction from DeepSeekMath:
https://arxiv.org/abs/2402.03300.

## Why Start Small

DeepSeekMath used a 7B model and 120B math-related tokens before GRPO post-
training: https://arxiv.org/abs/2402.03300. Kimi k1.5 focuses on long-context RL
scaling with large language models: https://arxiv.org/abs/2501.12599. DAPO,
GSPO, and CISPO are framed as large-scale LLM RL algorithms or systems:
https://arxiv.org/abs/2503.14476, https://arxiv.org/abs/2507.18071,
https://arxiv.org/abs/2506.13585.

Limes should therefore treat nanoGPT-scale RL as instrumentation practice:
define a verifiable reward, record seeds and artifacts, measure KL to a frozen
reference, and learn what breaks. The current command is:

```bash
./scripts/rl_toy_smoke.sh
```

It writes `out-rl/ppo_toy.json` with the reward curve, KL to the reference
policy, and clear non-claims.

## Feasible Now

- PPO-style toy loop: `rl/ppo_toy.py` implements a tiny character policy,
  exact/format reward, clipped policy update, value estimate, and KL tether to a
  frozen reference policy. The PPO shape follows the clipped-update and value
  function pattern documented by OpenAI Spinning Up:
  https://spinningup.openai.com/en/latest/algorithms/ppo.html.
- GRPO-style toy loop: feasible next as a grouped version of the same character
  task, with group-relative advantages and no value model, matching the
  DeepSeekMath GRPO motivation: https://arxiv.org/abs/2402.03300.
- MTP as an auxiliary training head: DeepSeek-V3 reports a multi-token
  prediction objective and speculative decoding use, but this repo should start
  with a gated auxiliary loss only, not an inference claim:
  https://arxiv.org/abs/2412.19437.
- BPB and artifact-size scoring: Parameter Golf evaluates small models under a
  16MB artifact constraint with tokenizer-agnostic bits per byte:
  https://github.com/openai/parameter-golf. `scripts/efficiency_score.py`
  computes BPB from token NLL and raw bytes, and records artifact size against a
  MiB limit.
- Tiny size lab config: `config/train_size_lab_16mb.py` is a conservative CPU
  config for generating small checkpoints that can be checked by
  `scripts/efficiency_score.py`.
- Low-rank, quantization, and parameter-tying experiments: Parameter Golf's
  challenge description explicitly points at constrained architectures and
  compression schemes such as aggressive parameter tying, low precision, and
  QAT: https://github.com/openai/parameter-golf.

## Not Appropriate Yet

- Full MoE training: DeepSeek-V3 is a 671B-total-parameter MoE with 37B active
  parameters per token, far outside this repo's scale:
  https://arxiv.org/abs/2412.19437.
- Large sparse or long-context attention systems: Kimi k1.5, MiniMax-M1, and the
  Chinese AI Research Catalog discuss long-context RL or hybrid/linear/sparse
  attention at scales that are not represented by this repo:
  https://arxiv.org/abs/2501.12599, https://arxiv.org/abs/2506.13585,
  https://www.himanshustwts.com/chinese-frontier/catalog.md.
- DAPO/GSPO/CISPO as production algorithms here: those sources target
  large-scale LLM RL systems, sequence-level MoE stability, or efficient
  long-CoT RL; they are useful references but should not be claimed as
  implemented by this nanoGPT lab:
  https://arxiv.org/abs/2503.14476, https://arxiv.org/abs/2507.18071,
  https://qwenlm.github.io/blog/gspo/, https://arxiv.org/abs/2506.13585.
- GLM-5.2 IndexShare / MTP-predicted-token KV reuse: a user note suggested that
  GLM-5.2 avoids reusing MTP predicted-token KV at `t+2` for efficiency and to
  reduce train/inference distribution shift. I did not find a primary source for
  this claim in this pass, so this repo should not present it as confirmed or
  implement against it.
- AutoResearch-RL as near-term roadmap: the arXiv page currently marks the
  submission withdrawn by arXiv Admin, so it is at most a cautionary pointer
  toward fixed evaluation environments and BPB-style rewards:
  https://arxiv.org/abs/2603.07300.

## Near-Term Acceptance Criteria

1. Add a GRPO toy next to `rl/ppo_toy.py` that samples groups per prompt and
   writes a comparable JSON artifact, citing https://arxiv.org/abs/2402.03300.
2. Add an MTP auxiliary-loss experiment only after the design in
   `docs/mtp-auxiliary-head-todo.md` is met, citing
   https://arxiv.org/abs/2412.19437.
3. Run a size-lab checkpoint with `config/train_size_lab_16mb.py`, then record
   raw-byte BPB and artifact MiB with `scripts/efficiency_score.py`, citing
   https://github.com/openai/parameter-golf.
