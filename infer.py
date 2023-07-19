from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model_ckpt = './nbs/training_article/checkpoint-46008'


    model = GPT2LMHeadModel.from_pretrained(model_ckpt)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_ckpt) 

    prompt = "áo khoác" # your input sentence
    input_ids = tokenizer("{}<|beginofdes|>".format(prompt), return_tensors="pt")['input_ids']

    max_length = 100
    gen_tokens = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.9,
            num_beams = 3,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id = 50259
        )
    
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)