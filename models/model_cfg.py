from dataclasses import dataclass

@dataclass
class GPT2Config:
    # Default config of GPT2-medium
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 24 
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

