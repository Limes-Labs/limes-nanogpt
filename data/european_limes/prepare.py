"""Prepare char-level dataset for european_limes corpus."""

import os
import pickle
import numpy as np

input_file = os.path.join(os.path.dirname(__file__), "input.txt")
with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

chars = sorted(list(set(data)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

print(f"chars: {len(chars)}")
print(f"train tokens: {len(train_data):,}")
print(f"val tokens: {len(val_data):,}")

def encode(s: str):
    return [stoi[c] for c in s]


train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

meta = {
    "vocab_size": len(chars),
    "itos": itos,
    "stoi": stoi,
    "char_level": True,
}
with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("wrote train.bin, val.bin, meta.pkl")