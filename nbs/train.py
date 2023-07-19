from transformers import pipeline, set_seed, GPT2LMHeadModel, AutoTokenizer, TextDataset, AutoModelForCausalLM
import datasets
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import math

from datasets import load_from_disk

# Load dataset 
train_dataset = load_from_disk("article_512/train")
test_dataset = load_from_disk("article_512/test")

model = AutoModelForCausalLM.from_pretrained("imthanhlv/vigpt2medium", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("imthanhlv/vigpt2medium")
special_tokens_dict = {'additional_special_tokens': ['<|beginofdes|>','<|endofdes|>', '<br>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

batch_size = 6
gradient_accumulation_steps = 2
num_gpu = 8

step_per_batch = math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps * num_gpu))
print('step_per_batch', step_per_batch)

def train(model):
    
    training_args = TrainingArguments(
        output_dir='training_article',
        # group_by_length=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=30,
        # fp16=True,
        save_steps=step_per_batch,
        eval_steps=step_per_batch,
        logging_steps=step_per_batch,
        learning_rate=1e-5,
        warmup_steps=step_per_batch * 5,
        save_total_limit=30,
        load_best_model_at_end=True,
        prediction_loss_only=True,
        metric_for_best_model='loss',
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience = 4)]
    )
    trainer.place_model_on_device = False
    trainer.train()
    trainer.save_model("training_article/best_model")
    tokenizer.save_pretrained("training_article/best_model")

train(model)