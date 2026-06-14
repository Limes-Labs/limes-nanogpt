"""
Minimal GPT training loop for Limes Labs.
Inspired by karpathy/nanoGPT.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPT, GPTConfig
from utils import resolve_device, device_type_from, amp_context, save_run_config

out_dir = "out-european-char"
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True
init_from = "scratch"
dataset = "european_limes"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 3e-5
device = "cuda"
dtype = "bfloat16"
compile = True

configurator = {}
exec(open("configurator.py").read())
import configurator as _  # noqa: F401

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = resolve_device(device)

torch.manual_seed(1337 + seed_offset)
device_type = device_type_from(device)
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype = "float16"
elif device_type == "mps":
    compile = False  # torch.compile unstable on MPS for small models
    dtype = "float16"

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = amp_context(device_type, dtype) if device_type != "cpu" else nullcontext()

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    save_run_config(
        out_dir,
        {
            "dataset": dataset,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "device": device,
            "dtype": dtype,
        },
    )

data_dir = os.path.join("data", dataset)
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    usable = len(data) - block_size - 1
    if usable <= 0:
        raise ValueError(
            f"{split} split too small ({len(data)} tokens) for block_size={block_size}. "
            f"Lower block_size in config or add data in data/{dataset}/"
        )
    ix = torch.randint(0, usable, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=True,
    vocab_size=None,
    dropout=dropout,
)

if init_from == "scratch":
    if meta_vocab_size is None:
        raise ValueError(f"run data/{dataset}/prepare.py first")
    model_args["vocab_size"] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args.update(checkpoint["model_args"])
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    for k in list(state_dict):
        if k.startswith("_orig_mod."):
            state_dict[k[10:]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

model.to(device)
raw_model = model
if compile and device_type == "cuda":
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


X, Y = get_batch("train")
t0 = time.time()
running_mfu = -1.0
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        mfu_str = f", mfu {running_mfu*100:.1f}%" if running_mfu >= 0 else ""
        print(f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}{mfu_str}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": {"dataset": dataset, "dtype": dtype},
                }
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if micro_step == gradient_accumulation_steps - 1:
            X, Y = get_batch("train")

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if device_type == "cuda" and iter_num > 10:
        running_mfu = raw_model.estimate_mfu(
            batch_size * gradient_accumulation_steps, dt
        )

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:e}")

    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    dist.destroy_process_group()