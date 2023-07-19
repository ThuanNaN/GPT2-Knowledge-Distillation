import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import datasets

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

# https://huggingface.co/docs/transformers/perplexity
def compute_ppl(model, encodings, device:str):
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


if __name__ == "__main__":
    device = "cuda"

    model_ckpt = './nbs/training_article/checkpoint-46008'

    model = GPT2LMHeadModel.from_pretrained(model_ckpt).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_ckpt) 

    special_tokens_dict = {'additional_special_tokens': ['<|beginofdes|>','<|endofdes|>', '<br>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    test_ds = get_text_ds('./data/fashion/raw/fashion_15_10_2022_train.txt')
    encodings = tokenizer(" ".join(test_ds["text"]), return_tensors="pt")

    ppl = compute_ppl(
        model,
        encodings,
        device
    )
    print(f"Perplexity: {ppl}")



