wandb_log = True 
wandb_project = 'mle'
wandb_run_name = 'train-student-adamw-drop02'

out_dir = 'out-shakespeare-char-baby'

dataset = 'shakespeare_char'

log_interval = 10 

eval_interval = 200 
eval_iters = 500
always_save_checkpoint = False


batch_size = 64
gradient_accumulation_steps = 1

block_size = 256 

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2


max_iters = 5000

learning_rate = 1e-3 
lr_decay_iters = 5000 
min_lr = 1e-4 
beta2 = 0.99 
warmup_iters = 200 
