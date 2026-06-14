"""Prepare char-level dataset from constitution excerpt."""
import pickle
from pathlib import Path

OUT = Path(__file__).resolve().parent
text = (OUT / "input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

import numpy as np

def encode(s):
    return [stoi[c] for c in s]

n = len(text)
train = np.array(encode(text[: int(n * 0.9)]), dtype=np.uint16)
val = np.array(encode(text[int(n * 0.9) :]), dtype=np.uint16)
train.tofile(OUT / "train.bin")
val.tofile(OUT / "val.bin")
with open(OUT / "meta.pkl", "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
print(f"constitution: {n} chars, vocab {len(chars)}")