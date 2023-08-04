import os
import pickle
import torch
from models import GPT, GPTConfig

def initialize_model(
    init_from: str = "scratch",
    ckpt_dir: str = "out",
    data_dir: str = "data",
    n_layer=24,
    n_head=16,
    n_embd=1024,
    block_size=1024,
    bias=False,
    dropout=0.0,
    device='cpu'
):

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout)
    checkpoint = None
    iter_num = 0
    best_val_loss = 1e9

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            # print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            pass
        # model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {ckpt_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

    return {
        "model": model,
        "model_args": model_args,
        "checkpoint": checkpoint,
        "resume_agrs": {
            "iter_num": iter_num,
            "best_val_loss": best_val_loss
        }
    }







