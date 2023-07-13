import torch
from tqdm import tqdm
from utils.data_processing import get_text_ds
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


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

    model = GPT2LMHeadModel.from_pretrained("imthanhlv/vigpt2medium").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("imthanhlv/vigpt2medium") 

    special_tokens_dict = {'additional_special_tokens': ['<|beginofdes|>','<|endofdes|>', '<br>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    test_ds = get_text_ds('./data/fashion/fashion_15_10_2022_test.txt')
    encodings = tokenizer("\n\n".join(test_ds["text"]), return_tensors="pt")

    ppl = compute_ppl(
        model,
        encodings,
        device
    )
    print(f"Perplexity: {ppl}")



