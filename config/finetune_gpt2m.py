wandb_log = False 
wandb_project = 'mle'
wandb_run_name = 'ft-teacher'

out_dir = 'out-shakespeare-char-gpt2m'

dataset = 'shakespeare_char'

log_interval = 10 

eval_interval = 20
eval_iters = 500
always_save_checkpoint = False

init_from = 'gpt2-medium'


batch_size = 4
gradient_accumulation_steps = 16

block_size = 1024 

max_iters = 500
learning_rate = 2e-5
decay_lr = False