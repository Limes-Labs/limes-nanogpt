#!/usr/bin/env python3
"""Report val perplexity for a checkpoint — used by eurobench smoke tests."""

import argparse
import math
import os
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig
from utils import resolve_device

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="out-european-char")
parser.add_argument("--dataset", default=None)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

device = resolve_device(args.device)
ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt["model_args"]))
sd = ckpt["model"]
for k in list(sd):
    if k.startswith("_orig_mod."):
        sd[k[10:]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

dataset = args.dataset or ckpt.get("config", {}).get("dataset", "european_limes")
meta = pickle.load(open(os.path.join("data", dataset, "meta.pkl"), "rb"))
block_size = ckpt["model_args"]["block_size"]
data = np.memmap(os.path.join("data", dataset, "val.bin"), dtype=np.uint16, mode="r")

losses = []
for _ in range(args.iters):
    ix = torch.randint(len(data) - block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)
    with torch.no_grad():
        _, loss = model(x, y)
    losses.append(loss.item())

mean_loss = sum(losses) / len(losses)
ppl = math.exp(mean_loss)
print(f"dataset={dataset} val_loss={mean_loss:.4f} perplexity={ppl:.2f}")