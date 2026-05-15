import time

# Fine-tune GPT-2 (124M) on TinyStories — adjust max_iters / batch for your hardware.
out_dir = "out-tinystories"
eval_interval = 50
eval_iters = 20
wandb_log = False
wandb_project = "tinystories"
wandb_run_name = "ft-tinystories-" + str(time.time())

dataset = "tinystories"
init_from = "gpt2"

always_save_checkpoint = False

# Micro-batch; effective tokens/step = batch_size * block_size * grad_accum
batch_size = 4
gradient_accumulation_steps = 8
block_size = 512
max_iters = 300

learning_rate = 3e-5
decay_lr = False
warmup_iters = 0
lr_decay_iters = max_iters
min_lr = learning_rate

dropout = 0.1
compile = False
