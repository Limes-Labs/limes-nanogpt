# limes-nanogpt

Minimal GPT training code for small, reproducible Limes Labs experiments.

This repo is intentionally not a frontier-model claim. It is a first technical
onboarding project: prepare a tiny legal corpus, train a toy character-level
model, produce an evaluation artifact, and document exactly what happened.

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT).

- Website: [limeslabs.eu](https://limeslabs.eu)
- Benchmark path: [eurobench](https://github.com/Limes-Labs/eurobench)
- Model card path: [model-card-template](https://github.com/Limes-Labs/model-card-template)
- Values path: [limes-constitution](https://github.com/Limes-Labs/limes-constitution)

## What This Is

- A readable nanoGPT-style training loop.
- A tiny Limes-authored starter corpus in `data/european_limes/input.txt`.
- CPU/Mac-friendly smoke and local configs.
- A perplexity eval hook that writes JSON for later EuroBench and model-card work.
- A place for contributors to learn the training/evaluation/artifact loop.

## What This Is Not

- Not a competitive language model.
- Not evidence of frontier capability.
- Not a safety benchmark.
- Not a multilingual benchmark.
- Not a production data pipeline.

## One-Command Smoke Test

On macOS or Linux with Python 3.11 or 3.12:

```bash
./scripts/smoke_test.sh
```

The script creates a versioned smoke venv such as `.venv-smoke-py311`, installs
`torch` and `numpy`, prepares the
tiny corpus, trains a small CPU checkpoint, samples text, and writes:

```text
out-smoke/run_config.json
out-smoke/ckpt.pt
out-smoke/eval.json
```

The generated `out-smoke/eval.json` is the reproducible artifact to attach to
experiment notes, EuroBench prototypes, and model-card drafts.

## Toy RL And Efficiency Lab

The repo now includes a deliberately tiny RL post-training path and an
efficiency scoring helper. These are workflow exercises, not model-quality
claims.

Run the PPO-style and GRPO-style toy RL smoke test:

```bash
./scripts/rl_toy_smoke.sh
```

It writes `out-rl/ppo_toy.json` and `out-rl/grpo_toy.json` with reward,
KL-to-reference, and exact-match metrics for tiny character policies. PPO uses
a scalar value estimate; GRPO uses grouped, critic-free reward normalization.

Compute tokenizer-agnostic BPB and an artifact-size check:

```bash
python3 scripts/efficiency_score.py \
  --artifact=out-smoke/ckpt.pt \
  --loss_nats=1.0 \
  --tokens=1000 \
  --raw_bytes=1000 \
  --max_artifact_mib=16 \
  --output_json=out-smoke/efficiency.json
```

Use real `loss_nats`, `tokens`, and `raw_bytes` from the same validation split
when recording an experiment. See
[docs/rl-and-efficiency-roadmap.md](docs/rl-and-efficiency-roadmap.md),
[docs/mtp-auxiliary-head-todo.md](docs/mtp-auxiliary-head-todo.md),
[docs/tokenizer-plan.md](docs/tokenizer-plan.md), and
[docs/optimizer-auxiliary-roadmap.md](docs/optimizer-auxiliary-roadmap.md).

If your default `python3` is newer than PyTorch supports, use:

```bash
PYTHON_BIN=python3.12 ./scripts/smoke_test.sh
```

## Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the tiny local dataset:

```bash
python3 data/european_limes/prepare.py
```

Run the tiny smoke config:

```bash
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

Run the slightly larger local config:

```bash
python3 train.py config/train_local_char.py --device=mps --compile=False
python3 eval_perplexity.py \
  --out_dir=out-local-char \
  --iters=25 \
  --batch_size=16 \
  --device=mps \
  --run_id=limes-nanogpt-local-char \
  --output_json=out-local-char/eval.json
```

Use `--device=cpu` if MPS/CUDA is unavailable.

## Data

The default dataset is tiny and committed in plain text:

```text
data/european_limes/input.txt
```

It is a Limes Labs starter text, not scraped web data. The prepare script builds
character-level `train.bin`, `val.bin`, and `meta.pkl` locally. Generated binary
files are ignored by git.

Optional constitution excerpts live in:

```text
data/constitution/input.txt
data/constitution_blend/prepare.py
```

They are for future alignment and documentation experiments, not a claim that
this repo trains aligned assistants today.

## Configs

- `config/train_smoke.py`: tiny CPU run for onboarding and CI-style checks.
- `config/train_local_char.py`: modest laptop run for local experiments.
- `config/train_european_char.py`: older larger char-level baseline.
- `config/train_constitution.py`: optional constitution blend experiment.
- `config/train_size_lab_16mb.py`: tiny CPU config for future BPB/artifact-size
  checks inspired by Parameter Golf-style constraints.

## Evaluation Artifact

`eval_perplexity.py` reports validation loss and perplexity. With
`--output_json`, it writes a small artifact with:

- run id and timestamp
- dataset name and split
- model args and checkpoint iteration
- `val_loss` and `perplexity`
- limitations that prevent overclaiming

This is the first bridge from laptop work to:

1. EuroBench task prototypes.
2. Model-card result tables.
3. Compute proposals that cite reproducible configs, logs, and artifacts.

## Tests

```bash
python3 -m unittest
```

The tests currently check the onboarding contract: eval help must expose JSON
artifact flags, the smoke scripts must run from the repository root, and the toy
RL/efficiency helpers must write reproducible JSON artifacts.

## Project Layout

```text
limes-nanogpt/
  model.py
  train.py
  sample.py
  eval_perplexity.py
  rl/
  docs/
  config/
  data/
  scripts/smoke_test.sh
  scripts/rl_toy_smoke.sh
  scripts/efficiency_score.py
  tests/
```

## Experiment Log

See [EXPERIMENTS.md](EXPERIMENTS.md) for the current smoke experiment record,
commands, artifacts, and non-claims.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first contributions include better
smoke assertions, clearer eval JSON fields for EuroBench, and tiny documented
dataset variants with license notes.

## License

MIT. See [LICENSE](LICENSE). Based on nanoGPT by Andrej Karpathy.
