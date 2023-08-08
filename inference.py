import torch
from transformers import GPT2LMHeadModel
from contextlib import nullcontext
import tiktoken
from utils import seed_everything

seed = 10
seed_everything(seed)


device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = './ckpt/teacher.pt'
ckpt = torch.load(ckpt_path)
model = GPT2LMHeadModel(ckpt['model_args'])
model.load_state_dict(ckpt['model'])
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start = "Would you proceed especially against Caius Marcius?"
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        y = model.generate(inputs = x, 
                            num_beams=4,
                            do_sample=True,
                            max_new_tokens=200)
        print(decode(y[0].tolist()))
        print('---------------')

