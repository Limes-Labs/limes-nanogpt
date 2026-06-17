#!/usr/bin/env bash
# CPU-only toy RL smoke test. No model-quality claim; writes a JSON run artifact.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_JSON="${OUTPUT_JSON:-out-rl/ppo_toy.json}"

"$PYTHON_BIN" rl/ppo_toy.py \
  --steps=12 \
  --batch_size=32 \
  --seed=1337 \
  --target=AB \
  --output_json="$OUTPUT_JSON"

echo "RL toy smoke OK"
