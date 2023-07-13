import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()


# add token again ? when load from our checkpoint ?
def load_tokenizer(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    special_tokens_dict = {'additional_special_tokens': ['<|beginofdes|>','<|endofdes|>', '<br>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# add token again ? when load from our checkpoint ?
def load_model(checkpoint_path: str):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(int(os.getenv("TOKENIZER_LEN")))
    return model
