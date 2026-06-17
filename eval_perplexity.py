#!/usr/bin/env python3
"""Report validation perplexity for a checkpoint and optional JSON artifact."""

import argparse
import json
import math
import os
import pickle
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-european-char")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path for a EuroBench/model-card friendly result artifact.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Stable identifier for this local run, stored in output_json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch

    from model import GPT, GPTConfig
    from utils import resolve_device

    device = resolve_device(args.device)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)

    dataset = args.dataset or ckpt.get("config", {}).get("dataset", "european_limes")
    meta_path = os.path.join("data", dataset, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    block_size = ckpt["model_args"]["block_size"]
    data = np.memmap(os.path.join("data", dataset, "val.bin"), dtype=np.uint16, mode="r")
    usable = len(data) - block_size - 1
    if usable <= 0:
        raise ValueError(
            f"val split too small ({len(data)} tokens) for block_size={block_size}; "
            "use a smaller smoke config or add data"
        )

    losses = []
    for _ in range(args.iters):
        ix = torch.randint(0, usable, (args.batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        ).to(device)
        y = torch.stack(
            [
                torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
                for i in ix
            ]
        ).to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    run_id = args.run_id or os.path.basename(os.path.abspath(args.out_dir))
    result = {
        "schema_version": "limes-eval-v0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_family": "eurobench-smoke",
        "task": "char_level_validation_perplexity",
        "dataset": {
            "name": dataset,
            "split": "val",
            "vocab_size": meta["vocab_size"],
            "char_level": True,
        },
        "metrics": {
            "val_loss": mean_loss,
            "perplexity": ppl,
        },
        "evaluation": {
            "iters": args.iters,
            "batch_size": args.batch_size,
            "device": device,
        },
        "model": {
            "checkpoint": ckpt_path,
            "iteration": ckpt.get("iter_num"),
            "model_args": ckpt["model_args"],
        },
        "limitations": [
            "Character-level validation perplexity on a tiny local corpus.",
            "Not a capability, safety, multilinguality, or frontier-model claim.",
        ],
    }

    print(f"dataset={dataset} val_loss={mean_loss:.4f} perplexity={ppl:.2f}")
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
