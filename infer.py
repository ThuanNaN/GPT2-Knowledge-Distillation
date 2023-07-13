from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = GPT2LMHeadModel.from_pretrained("imthanhlv/vigpt2medium").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("imthanhlv/vigpt2medium") 

    prompt = "áo sơ mi" # your input sentence
    input_ids = tokenizer("{}<|beginofdes|>".format(prompt), return_tensors="pt")['input_ids'].to(device)

    max_length = 100
    gen_tokens = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.9,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)