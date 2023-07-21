wandb_log = True 
wandb_project = 'mle'
wandb_run_name = 'ft-teacher-adamw-drop01'

out_dir = 'shakespeare-char-gpt2m'

dataset = 'shakespeare_char'

log_interval = 10 

eval_interval = 50
eval_iters = 500
always_save_checkpoint = False

init_from = 'gpt2-medium'
dropout = 0.1

batch_size = 4
gradient_accumulation_steps = 16


max_iters = 500
learning_rate = 1e-4
decay_lr = False