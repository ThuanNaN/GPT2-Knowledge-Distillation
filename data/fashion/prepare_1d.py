import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets
from dotenv import load_dotenv
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("imthanhlv/vigpt2medium")
special_tokens_dict = {'additional_special_tokens': ['<|beginofdes|>','<|endofdes|>', '<br>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token

context_length = int(os.getenv("CONTEXT_LEN"))
num_proc = 4

def get_text_ds(path):
    lines = []
    text = ''
    with open(path) as f:
        for line in f:
            if line.find('xxxxxEndxxxxx') != -1:
                if text.strip():
                    lines.append(text.strip())
                    text = ''
                else:
                    continue
            else:
                text = text + line
    vi_ds = datasets.Dataset.from_dict({'text': lines})
    return vi_ds

def tokenize_function(batch):
    outputs = tokenizer(
        batch['text'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding='max_length'
    )["input_ids"][0]
    result = {
        "ids": outputs,
        "len": len(outputs)
    }
    return result

if __name__ == "__main__":
    train_path = './raw/fashion_15_10_2022_train.txt'
    test_path = './raw/fashion_15_10_2022_test.txt'

    tokenized = datasets.DatasetDict(
        {
            'train': get_text_ds(train_path),
            'val': get_text_ds(test_path)
        }
    ).map(
        tokenize_function,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    print('tokenization finished')
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        filename = Path(".") / f"{split}.bin"
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()

        