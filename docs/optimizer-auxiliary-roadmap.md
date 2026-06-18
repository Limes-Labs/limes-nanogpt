# Optimizer And Auxiliary-Head Roadmap

This document extracts public-safe research directions from private Limes Labs
audits. It should guide new experiments without copying private repositories,
claims, or implementation details into this public repo.

## Public-Safe Rule

Do not copy private code wholesale. Any public extraction needs a clean-room
rewrite, attribution review, license check, secret scan, and a smaller public
claim than the private notes may suggest. The first public artifact should be a
testable experiment proposal, not a polished claim.

## Tracks

### Sekant/SCTR Optimizer Probes

Sekant/SCTR should become an optimizer comparison track only after the baseline
contract is stable:

- Baseline first: compare against AdamW on the same seed, token budget, update
  count, and wall-clock class.
- Tiny transfer gate: an idea must improve a toy objective and one tiny
  language-model run before it gets larger compute.
- Negative-result log: record failures and instability, not only wins.
- Cost accounting: report extra optimizer state, extra matmuls, and update
  overhead beside validation loss or bits per byte.

Acceptance tests:

- A future optimizer harness runs on CPU with a tiny synthetic objective.
- AdamW remains the required baseline.
- The result JSON records optimizer name, seed, updates, tokens, elapsed
  seconds, and parameter/state bytes.

### Robust Optimizer Stability

The optimizer stability workstream should stress noisy and heavy-tail gradients
before touching real training runs:

- Inject outlier gradients in a tiny convex or low-dimensional toy objective.
- Track gradient norm, update norm, loss spikes, and recovery time.
- Compare clipping, damping, and robust update rules under the same update
  budget.
- Treat stability as useful only if it survives a fixed compute budget.

Acceptance tests:

- A deterministic toy test shows the harness can create a controlled outlier.
- The reported metrics distinguish raw gradient norm from applied update norm.
- The experiment can finish in seconds on CPU.

### Multi-State And Future-Token Auxiliary Heads

Multi-State/future-token auxiliary heads are the public label for the
multi-state and future-token ideas from the audit. They belong in this repo as
optional auxiliary losses, not as inference-speed claims. They connect to the
existing `docs/mtp-auxiliary-head-todo.md` plan:

- Keep next-token loss unchanged by default.
- Add future-token heads only behind explicit config flags.
- Mask positions without future targets.
- Report auxiliary-loss weight separately from the main loss.
- Preserve generation behavior until a separate speculative decoding benchmark
  exists.

Acceptance tests:

- `mtp_depth = 0` preserves the current `GPT.forward(idx, targets)` behavior.
- `mtp_depth > 0` adds a measurable auxiliary loss on a tiny CPU batch.
- Generated text is unchanged by the training-only auxiliary head path.

### Update-Budget Accounting

Every efficiency experiment should charge the resources it consumes:

- tokens processed
- optimizer updates
- examples and raw bytes
- parameter bytes and optimizer-state bytes
- auxiliary-head parameters
- selection, teacher, critic, or distillation compute
- wall-clock seconds on the reported hardware class

This matters especially when comparing PPO, GRPO, OPD, OPSD, optimizer variants,
and auxiliary heads. A smaller validation loss is not an efficiency result if it
used hidden teacher, critic, selector, or search compute that was not charged.

## Near-Term Tasks

1. Add a tiny optimizer harness with AdamW and a placeholder custom optimizer
   interface.
2. Add result JSON fields for update-budget accounting.
3. Implement MTP/future-token auxiliary heads only after the acceptance tests in
   `docs/mtp-auxiliary-head-todo.md` are met.
4. Add a public experiment template that records negative, mixed, and diagnostic
   outcomes without implying deployment readiness.
