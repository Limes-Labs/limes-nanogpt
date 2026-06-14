# limes-nanogpt

Minimal GPT trainer for **small, reproducible European language experiments**.

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT). Limes Labs maintains this fork for transparent open work: train tiny models on European text, publish configs, log failures, and scale only when compute is real.

> We are not claiming frontier models. We are organizing capability.

- Website: [limeslabs.eu](https://limeslabs.eu)
- Benchmarks: [eurobench](https://github.com/Limes-Labs/eurobench)
- Join: [limeslabs.eu/join](https://limeslabs.eu/join)

## Why this exists

nanoGPT is the best minimal reference for GPT training. Karpathy now points to [nanochat](https://github.com/karpathy/nanochat) for newer work, but nanoGPT remains ideal for **learning, hacking, and small runs**.

Limes Labs adds:

- European starter corpus (`data/european_limes/`)
- CPU/MPS-friendly default config
- Explicit links to EuroHPC / IT4LIA compute path (see [compute-access-notes](https://github.com/Limes-Labs/compute-access-notes))
- Integration target for [eurobench](https://github.com/Limes-Labs/eurobench) evaluation

## Quick start (Mac / laptop)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data/european_limes/prepare.py
python train.py config/train_european_char.py
python sample.py --out_dir=out-european-char
```

On Apple Silicon, add `--device=mps`. On CPU only:

```bash
python train.py config/train_european_char.py --device=cpu --compile=False --max_iters=1500
```

## Project layout

```
limes-nanogpt/
├── model.py              # GPT definition (nanoGPT lineage)
├── train.py              # Training loop
├── sample.py             # Text generation
├── configurator.py       # Config overrides
├── config/
│   └── train_european_char.py
└── data/european_limes/
    ├── prepare.py
    └── input.txt         # Starter European manifesto text
```

## Roadmap

- [ ] BPE tokenizer path for multilingual European data
- [ ] Export checkpoints compatible with eurobench eval harness
- [ ] Document GPU-hour estimates for IT4LIA / AI Factory runs
- [ ] Fine-tune from open-weight European base models (not just char-level)

## License

MIT — see [LICENSE](LICENSE). Based on nanoGPT by Andrej Karpathy.

## Attribution

If you use this repo in public work, cite Limes Labs and link to the training config and dataset source you used.