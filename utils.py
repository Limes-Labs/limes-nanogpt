"""Training utilities for limes-nanogpt."""

import os
import json
import torch


def resolve_device(requested: str) -> str:
    """Pick best available device; honour explicit cpu/mps/cuda."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested in ("cuda", "mps"):
        # fallback when requested accelerator missing
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type_from(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def amp_context(device_type: str, dtype: str):
    from contextlib import nullcontext

    if device_type == "cpu":
        return nullcontext()
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    return torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def save_run_config(out_dir: str, config: dict) -> None:
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)