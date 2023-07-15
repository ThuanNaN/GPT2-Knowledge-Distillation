import os
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
        # padding='max_length'
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        input_batch.append(input_ids)
    result = {}
    result['input_ids'] = input_batch
    # result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":
    train_path = './raw/fashion_15_10_2022_train.txt'
    test_path = './raw/fashion_15_10_2022_test.txt'

    train_ds = get_text_ds(train_path)
    test_ds = get_text_ds(test_path)

    tokenized_train_datasets = train_ds.map(
        tokenize_function, batched=True, remove_columns=['text']
    )
    tokenized_test_datasets = test_ds.map(
        tokenize_function, batched=True, remove_columns=['text']
    )
    
    train_data_path = 'processed/train'
    tokenized_train_datasets.save_to_disk(train_data_path)

    test_data_path = 'processed/test'
    tokenized_test_datasets.save_to_disk(test_data_path)

