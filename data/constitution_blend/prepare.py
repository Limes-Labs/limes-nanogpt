"""Build constitution_blend dataset from european_limes + constitution excerpts."""

import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ROOT / "european_limes" / "input.txt",
    ROOT / "constitution" / "input.txt",
]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def main():
    chunks = []
    for src in SOURCES:
        if not src.exists():
            raise FileNotFoundError(f"missing {src} — run each prepare.py first")
        chunks.append(src.read_text(encoding="utf-8"))
    data = "\n\n---\n\n".join(chunks)
    (OUT / "input.txt").write_text(data, encoding="utf-8")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    import numpy as np

    def encode(s):
        return [stoi[c] for c in s]

    n = len(data)
    train_ids = np.array(encode(data[: int(n * 0.9)]), dtype=np.uint16)
    val_ids = np.array(encode(data[int(n * 0.9) :]), dtype=np.uint16)
    train_ids.tofile(OUT / "train.bin")
    val_ids.tofile(OUT / "val.bin")
    with open(OUT / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)
    print(f"constitution_blend: {n:,} chars, vocab {vocab_size}")


if __name__ == "__main__":
    main()