# Tiny CPU smoke run for contributor onboarding.
# Goal: produce a checkpoint and eval artifact quickly, not a useful model.

out_dir = "out-smoke"
dataset = "european_limes"

eval_interval = 10
eval_iters = 5
log_interval = 5
always_save_checkpoint = True

gradient_accumulation_steps = 1
batch_size = 16
block_size = 64
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0

learning_rate = 1e-3
max_iters = 20
lr_decay_iters = 20
min_lr = 1e-4
warmup_iters = 2

device = "cpu"
dtype = "float32"
compile = False
