import os
import argparse
import time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import math
import numpy as np
from models.utils import initialize_model
from utils import seed_everything

seed_everything(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

rho = 0.05
interval = 10
hess_interval = interval

# learning rate decay scheduler (cosine with warmup)
def get_lr(iter, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# poor man's data loader
def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, dataloaders, eval_iters, **kwargs):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = dataloaders[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, **kwargs)
            with ctx:
                outputs = model(X, Y)
            losses[k] = outputs['loss'].item()
        out[split] = losses.mean()
    model.train()
    return out

def train(opt, dataloaders, model, optimizer, iter_num, best_val_loss, dtype):
    if opt.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0
    
    lr_decay_args = {
        "learning_rate": opt.learning_rate,
        "warmup_iters": opt.warmup_iters,
        "lr_decay_iters": opt.lr_decay_iters,
        "min_lr": opt.min_lr
    }
    get_batch_args = {
        "batch_size": opt.batch_size , 
        "block_size": opt.block_size, 
    }
    total_bs = opt.accumulation_steps * opt.batch_size
    X, Y = get_batch(dataloaders['train'], **get_batch_args) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    running_mfu = -1.0
    num_param = 1
    num_effective = 0
    momentum_norm = 0
    hessian_norm = 0
    hessian_norm2 = 0
    clip_time = 0
    while True:
        lr = get_lr(iter_num, **lr_decay_args) if opt.decay_lr else opt.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % opt.eval_interval == 0:
            losses = estimate_loss(model, dataloaders, opt.eval_iters, **get_batch_args)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if opt.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                })
            if losses['val'] < best_val_loss or opt.always_save_ckpt:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': opt.model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': opt,
                    }
                    print(f"saving checkpoint to {opt.save_dir}")
                    torch.save(checkpoint, os.path.join(opt.save_dir, 'ckpt.pt'))
    
        if iter_num == 0 and opt.eval_only:
            break
        if iter_num % hess_interval != hess_interval - 1:
            for _ in range(opt.accumulation_steps):
                with ctx:
                    outputs = model(X, Y)
                    loss = outputs['loss'] / opt.accumulation_steps
                X, Y = get_batch(dataloaders['train'], **get_batch_args)
                (loss / opt.accumulation_steps).backward()
            if opt.grad_clip != 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                if total_norm.item() > opt.grad_clip:
                    clip_time += 1
            optimizer.step(bs= total_bs* opt.block_size)
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % opt.log_interval == 0:
                lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(opt.batch_size * opt.accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
                total_param_norm = 0
                momentum_norm = 0
                params = []
                for (name, p) in model.named_parameters():
                    params.append(p)
                for p in params:
                    param_norm = p.data.norm(2)
                    total_param_norm += param_norm.item() ** 2
                total_param_norm = total_param_norm ** 0.5
                momentum_norm = 0
                LL = len(optimizer.state_dict()['state'])
                for jj in range(LL):
                    momentum_norm += (optimizer.state_dict()['state'][jj]['exp_avg'].detach().norm(2)) ** 2
                momentum_norm = torch.sqrt(momentum_norm).item()
                if opt.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "param_norm": total_param_norm,
                        "momentum_norm" : momentum_norm,
                        "hessian_norm": hessian_norm,
                        "hessian_norm2": hessian_norm2,
                        "train/win_rate": num_effective / num_param,
                        "train/clip_rate": clip_time / (iter_num + 1)
                        
                    }, step=iter_num)
            iter_num += 1
            local_iter_num += 1
        else:
            for _ in range( opt.accumulation_steps):
                with ctx:
                    outputs = model(X, Y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch(dataloaders['train'], **get_batch_args)
                # backward pass, with gradient scaling if training in fp16
                (outputs['loss'] /  opt.accumulation_steps).backward()
            # clip the gradient
            if opt.grad_clip != 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                if total_norm.item() > opt.grad_clip:
                    clip_time += 1
            # step the optimizer and scaler if training in fp16
            optimizer.step(bs=total_bs * opt.block_size)
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % opt.log_interval == 0 :
                lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(opt.batch_size *  opt.accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

                total_param_norm = 0
                momentum_norm = 0
                    
                if opt.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "param_norm": total_param_norm,
                        "momentum_norm" : momentum_norm,
                        "hessian_norm": hessian_norm.item(),
                        "hessian_norm2": hessian_norm2,
                        "train/win_rate": num_effective / num_param,
                        "train/clip_rate": clip_time / (iter_num + 1)
                        
                    }, step=iter_num)
            iter_num += 1
            local_iter_num += 1

            for _ in range(opt.accumulation_steps):
                with ctx:
                    outputs = model(X, 0)
                X, Y = get_batch(dataloaders['train'], **get_batch_args)
                samp_dist = torch.distributions.Categorical(logits=outputs['logits'])
                y_sample = samp_dist.sample()
                loss = F.cross_entropy(outputs['logits'].view(-1, outputs['logits'].size(-1)), y_sample.view(-1), ignore_index=-1)
                # backward pass, with gradient scaling if training in fp16
                (loss / opt.accumulation_steps).backward()
            # clip the gradient
            if opt.grad_clip != 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            # step the optimizer and scaler if training in fp16
            optimizer.update_hessian()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            num_param = 0
            num_effective = 0
            hessian_norm = 0
            hessian_norm2 = 0
            
            LL = len(optimizer.state_dict()['state'])
            
            for jj in range(LL):
                num_param += optimizer.state_dict()['state'][jj]['exp_avg'].numel()
                num_effective += torch.sum(torch.abs(optimizer.state_dict()['state'][jj]['exp_avg']) < rho * total_bs * opt.block_size * optimizer.state_dict()['state'][jj]['hessian'])
                hessian_norm += optimizer.state_dict()['state'][jj]['hessian'].detach().norm(1).item()
                hessian_norm2 += optimizer.state_dict()['state'][jj]['hessian'].detach().norm(2).item() ** 2
            hessian_norm2 = hessian_norm2 ** 0.5

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if True:
                # loss as float. note: this is a CPU-GPU sync point
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(opt.batch_size *  opt.accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")


        # termination conditions
        if iter_num > opt.max_iters:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-log', action='store_true', help='Log result into WanDB')
    parser.add_argument('--wandb-project', type=str, default="gpt2-medium", help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default="gpt2m-init", help='WandB run name')

    parser.add_argument('--save-dir', type=str, default="output", help='place to save checkpoint')
    parser.add_argument('--dataset', type=str, default="shakespeare_char", help='dataset folder')

    parser.add_argument('--batch-size', type=int, default=32, help='if gradient_accumulation_steps > 1, this is the micro-batch size')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='used to simulate larger batch sizes')
    parser.add_argument('--block-size', type=int, default=1024, help='context length')
    # model config
    parser.add_argument('--init-from', type=str, default="scratch", help="'scratch' or 'resume' or 'gpt2*")
    parser.add_argument('--num-layer', type=int, default=24, help='number of transformer layer')
    parser.add_argument('--num-head', type=int, default=16, help='number of head in multi-head')
    parser.add_argument('--num-embd', type=int, default=1024, help='number of embedding dim')
    parser.add_argument('--dropout', type=float, default=0.0, help='for pretraining 0 is good, for finetuning try 0.1+')
    parser.add_argument('--bias', action='store_true', help='do we use bias inside LayerNorm and Linear layers?')
    # adamw optimizer
    parser.add_argument('--max-iters', type=int, default=50000, help='number of interations training')
    parser.add_argument('--learning-rate', type=float, default=6e-4, help='max learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-1, help='weight decay for optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='')
    parser.add_argument('--beta2', type=float, default=0.95, help='')
    parser.add_argument('--rho', type=float, default=0.03, help='')

    parser.add_argument('--grad-clip', type=float, default=1.0, help='clip gradients at this value, or disable if == 0.0')
    # learning rate decay settings
    parser.add_argument('--decay-lr', action='store_true', help='whether to decay the learning rate')
    parser.add_argument('--warmup-iters', type=int, default=2000, help='how many steps to warm up for')
    parser.add_argument('--lr-decay-iters', type=int, default=50000, help='should be ~= max_iters per Chinchilla')
    parser.add_argument('--min-lr', type=float, default=6e-5, help='minimum learning rate, \
                                                                    should be ~= learning_rate/10 per Chinchilla')
    #eval and log
    parser.add_argument('--eval-interval', type=int, default=2000, help='')
    parser.add_argument('--log-interval', type=int, default=1, help='')
    parser.add_argument('--eval-iters', type=int, default=200, help='number of iters using evaluation')
    parser.add_argument('--eval-only', action='store_true', help='if True, script exits right after the first eval')
    parser.add_argument('--always-save-ckpt', action='store_true', help='if True, always save a checkpoint after each eval')
    parser.add_argument('--compile', action='store_true', help='use to compile torch model, required torch>=2.0.1')
    opt = parser.parse_args()

    # logging
    if opt.wandb_log:
        import wandb
        wandb.init(project=opt.wandb_project, name=opt.wandb_run_name, config=opt)
    os.makedirs(opt.save_dir, exist_ok=True)

    tokens_per_iter = opt.accumulation_steps * opt.batch_size * opt.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    data_dir = os.path.join('data', opt.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    dataloaders = {
        "train": train_data,
        "val": val_data
    }
    model_cfg = {
        "n_layer": opt.num_layer, 
        "n_head": opt.num_head,
        "n_embd": opt.num_embd,
        "block_size": opt.block_size,
        "bias": opt.bias,
        "dropout": opt.dropout
    }
    initialized = initialize_model(init_from = opt.init_from,
                                    ckpt_dir = opt.save_dir,
                                    data_dir = opt.dataset,
                                    device = device,
                                    **model_cfg)
    
    model = initialized['model']
    opt.model_args = initialized['model_args']
    iter_num = initialized['resume_agrs']['iter_num']
    best_val_loss = initialized['resume_agrs']['best_val_loss']

    # crop down the model block size if desired, using model surgery
    if opt.block_size < model.config.block_size:
        model.crop_block_size(opt.block_size)
        opt.model_args['block_size'] = opt.block_size # so that the checkpoint will have the right value
    model.to(device)

    # optimizer
    optimizer = model.configure_optimizers("sophiag", opt.weight_decay, opt.learning_rate, (opt.beta1, opt.beta2), opt.rho, device_type)
    if opt.init_from == 'resume':
        optimizer.load_state_dict(initialized['checkpoint']['optimizer'])
    checkpoint = None # free up memory

    train(opt, dataloaders, model, optimizer, iter_num, best_val_loss, dtype)
