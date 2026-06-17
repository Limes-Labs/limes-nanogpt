#!/usr/bin/env bash
# Quick smoke test: CPU-friendly and intended to finish in minutes on a laptop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
  else
    PYTHON_BIN=python3
  fi
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
VENV_DIR=".venv-smoke-py${PYTHON_VERSION}"

"$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null || true
source "$VENV_DIR/bin/activate"
pip install -q -r requirements.txt

"$PYTHON_BIN" data/european_limes/prepare.py
"$PYTHON_BIN" train.py config/train_smoke.py

"$PYTHON_BIN" eval_perplexity.py \
  --out_dir=out-smoke \
  --iters=5 \
  --batch_size=8 \
  --device=cpu \
  --run_id=limes-nanogpt-smoke \
  --output_json=out-smoke/eval.json

"$PYTHON_BIN" sample.py \
  --out_dir=out-smoke \
  --device=cpu \
  --num_samples=1 \
  --max_new_tokens=80
echo "Smoke test OK"
