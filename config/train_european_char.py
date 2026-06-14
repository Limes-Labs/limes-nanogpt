# European char-level GPT — laptop-friendly defaults
out_dir = "out-european-char"
eval_interval = 200
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

dataset = "european_limes"
gradient_accumulation_steps = 1
batch_size = 48
block_size = 128  # small char corpus; 256 overflows val split

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 3e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 3e-5
warmup_iters = 100

device = "cuda"
compile = True