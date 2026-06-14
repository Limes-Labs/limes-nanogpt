#!/usr/bin/env bash
# Quick smoke test — CPU-friendly, ~2 min on laptop
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv-smoke 2>/dev/null || true
source .venv-smoke/bin/activate
pip install -q -r requirements.txt

python data/european_limes/prepare.py
python train.py config/train_european_char.py \
  --device=cpu \
  --compile=False \
  --max_iters=50 \
  --eval_interval=25 \
  --log_interval=10 \
  --out_dir=out-smoke

python sample.py --out_dir=out-smoke --max_new_tokens=80
echo "Smoke test OK"