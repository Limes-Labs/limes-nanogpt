# Slightly larger local baseline for CPU/MPS laptops.
# This is still a toy char-level model; publish config and eval JSON with results.

out_dir = "out-local-char"
dataset = "european_limes"

eval_interval = 100
eval_iters = 25
log_interval = 10
always_save_checkpoint = True

gradient_accumulation_steps = 1
batch_size = 32
block_size = 128
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 6e-4
max_iters = 500
lr_decay_iters = 500
min_lr = 6e-5
warmup_iters = 20

device = "cuda"
dtype = "bfloat16"
compile = True
