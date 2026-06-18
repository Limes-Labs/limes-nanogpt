#!/usr/bin/env bash
# CPU-only toy RL smoke test. No model-quality claim; writes JSON run artifacts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PPO_OUTPUT_JSON="${PPO_OUTPUT_JSON:-out-rl/ppo_toy.json}"
GRPO_OUTPUT_JSON="${GRPO_OUTPUT_JSON:-out-rl/grpo_toy.json}"

"$PYTHON_BIN" rl/ppo_toy.py \
  --steps=12 \
  --batch_size=32 \
  --seed=1337 \
  --target=AB \
  --output_json="$PPO_OUTPUT_JSON"

"$PYTHON_BIN" rl/grpo_toy.py \
  --steps=12 \
  --groups=8 \
  --group_size=4 \
  --seed=1337 \
  --target=AB \
  --output_json="$GRPO_OUTPUT_JSON"

echo "RL toy smoke OK"
