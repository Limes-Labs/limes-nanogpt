# Constitution + European manifesto char-level run
out_dir = "out-constitution-char"
dataset = "constitution_blend"

eval_interval = 150
eval_iters = 80
log_interval = 10
always_save_checkpoint = True

batch_size = 48
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.12

learning_rate = 3e-4
max_iters = 2500
lr_decay_iters = 2500
min_lr = 3e-5
warmup_iters = 80

device = "cuda"
compile = True