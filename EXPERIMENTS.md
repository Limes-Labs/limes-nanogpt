# Experiments

This log records small reproducible experiments. It should make failures and
limits as visible as results.

## Smoke Run: `limes-nanogpt-smoke`

Purpose: prove that a new contributor can prepare data, train a tiny checkpoint,
run evaluation, and produce a JSON artifact on a laptop.

Commands:

```bash
./scripts/smoke_test.sh
```

Equivalent manual commands:

```bash
python3 data/european_limes/prepare.py
python3 train.py config/train_smoke.py
python3 eval_perplexity.py \
  --out_dir=out-smoke \
  --iters=5 \
  --batch_size=8 \
  --device=cpu \
  --run_id=limes-nanogpt-smoke \
  --output_json=out-smoke/eval.json
python3 sample.py --out_dir=out-smoke --device=cpu --num_samples=1
```

Expected artifacts:

```text
out-smoke/run_config.json
out-smoke/ckpt.pt
out-smoke/eval.json
```

What to record after running:

```text
date:
machine:
python:
torch:
device:
duration:
val_loss:
perplexity:
notes:
```

## Non-Claims

- The smoke model is not useful for downstream tasks.
- Character-level perplexity on this tiny corpus is not a EuroBench score.
- Generated text is expected to be noisy.
- Results are for workflow verification and contributor learning.

## Path To Larger Work

1. Laptop experiment: config, commit hash, generated data checksum, eval JSON.
2. EuroBench prototype: map `eval.json` fields into a task result record.
3. Model card: copy metrics, dataset notes, intended use, and limitations.
4. Compute proposal: cite the reproducible trail and explain what scale would
   test next.

Before asking for more compute, add a new section here with the exact config,
data source, artifact path, and the question the run answers.
