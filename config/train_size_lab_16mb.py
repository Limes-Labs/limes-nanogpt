# Tiny CPU config for future Parameter Golf-style size/BPB experiments.
# Goal: keep checkpoints comfortably below a 16 MiB artifact limit, then compare
# BPB on the same raw-byte corpus. This is not a quality target.

out_dir = "out-size-lab-16mb"
dataset = "european_limes"

eval_interval = 10
eval_iters = 5
log_interval = 5
always_save_checkpoint = True

gradient_accumulation_steps = 1
batch_size = 16
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 30
lr_decay_iters = 30
min_lr = 1e-4
warmup_iters = 2

device = "cpu"
dtype = "float32"
compile = False
