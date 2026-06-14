"""Generate text from a trained limes-nanogpt checkpoint."""

import os
import pickle
import argparse

import torch

from model import GPT, GPTConfig
from utils import resolve_device

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="out-european-char")
parser.add_argument("--start", type=str, default="\n")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--max_new_tokens", type=int, default=300)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=200)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

device = resolve_device(args.device)

ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
assert os.path.exists(ckpt_path), f"no checkpoint at {ckpt_path}"

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
for k in list(state_dict):
    if k.startswith("_orig_mod."):
        state_dict[k[10:]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

run_cfg_path = os.path.join(args.out_dir, "run_config.json")
dataset = checkpoint.get("config", {}).get("dataset", "european_limes")
if os.path.exists(run_cfg_path):
    import json

    with open(run_cfg_path, encoding="utf-8") as f:
        dataset = json.load(f).get("dataset", dataset)

meta_path = os.path.join("data", dataset, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

start_ids = encode(args.start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

for k in range(args.num_samples):
    torch.manual_seed(args.seed + k)
    y = model.generate(
        x,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(decode(y[0].tolist()))
    print("---------------")