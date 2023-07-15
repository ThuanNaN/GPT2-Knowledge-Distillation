import math
from pathlib import Path
import torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
from utils import load_tokenizer, load_model_from_pretrain, logger
import wandb
wandb.login()

ROOT = Path(".")
TEACHER_DIR = ROOT / "ft_teacher"
SAVE_DIR = TEACHER_DIR / "adamw"

pretrain_name = "imthanhlv/vigpt2medium"
model = load_model_from_pretrain(pretrain_name)
tokenizer = load_tokenizer(pretrain_name)

# Load dataset 
train_dataset = load_from_disk("./data/fashion/processed/train")
test_dataset = load_from_disk("./data/fashion/processed/test")

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

num_epochs = 15
batch_size = 4
lr = 5e-3
gradient_accumulation_steps = 4
num_gpu = 1

step_per_batch = math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps * num_gpu))
logger.info(f"Step per batch: {step_per_batch}")




def train(model):
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=step_per_batch,
        save_total_limit=5,
        prediction_loss_only=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=step_per_batch * 5,

        evaluation_strategy="steps",
        eval_steps=step_per_batch,
        logging_steps=step_per_batch,
        learning_rate=lr,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        optim='adamw_torch',
        report_to="wandb",
    )
    wandb.init(
        project="distill-gpt2m",
        name="ft_teacher_simple_adamw",
        config=training_args
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    trainer.place_model_on_device = False
    trainer.train()
    trainer.save_model(SAVE_DIR/ "best_model")
    tokenizer.save_pretrained(SAVE_DIR / "best_model")

    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")


train(model)
