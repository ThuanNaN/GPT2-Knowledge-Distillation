import os
from logger import get_logger
import datasets
from pretrain_loader import load_tokenizer
from dotenv import load_dotenv
load_dotenv()
logger = get_logger("dataloader")

logger.info("Load tokenizer")
tokenizer = load_tokenizer("imthanhlv/vigpt2medium")

context_length = int(os.getenv("CONTEXT_LEN"))
logger.info(f"Context length: {context_length}")

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


def tokenize_function(sample):
    outputs = tokenizer(
        sample['text'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True
        #padding='max_length' # >> Fixe-length when training
  )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        input_batch.append(input_ids)
    result = {}
    result['input_ids'] = input_batch
    # result["labels"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":
    train_path = './data/fashion/raw/fashion_15_10_2022_train.txt'
    test_path = './data/fashion/raw/fashion_15_10_2022_test.txt'

    logger.info("Load raw dataset")
    train_dataset = get_text_ds(train_path)
    test_dataset = get_text_ds(test_path)

    logger.info("Tokenize train dataset")
    tokenized_train_datasets = train_dataset.map(
        tokenize_function, batched=True, remove_columns=['text']
    )
    logger.info("Tokenize test dataset")
    tokenized_test_datasets = test_dataset.map(
        tokenize_function, batched=True, remove_columns=['text']
    )

    train_data_path = f'./data/fashion/processed/article_{context_length}/train'
    test_data_path = f'./data/fashion/processed/article_{context_length}/test'

    logger.info(f"Train dataset path: {train_data_path}")
    logger.info(f"Test dataset path: {test_data_path}")

    tokenized_train_datasets.save_to_disk(train_data_path)
    tokenized_test_datasets.save_to_disk(test_data_path)


