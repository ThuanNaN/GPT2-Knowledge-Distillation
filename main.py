import yaml
from yaml import SafeLoader
import math
from dotenv import load_dotenv
load_dotenv()
import torch
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from utils.pretrain_loader import load_tokenizer, load_model
from utils.logger import get_logger
LOGGER = get_logger("Training")


if __name__ == "__main__":

    LOGGER.info("Start load dataset from disk")
    data_dir = "./data/fashion/processed/article_512"
    train_dataset = load_from_disk(data_dir+"/train")
    test_dataset = load_from_disk(data_dir+"/test")
    LOGGER.info("Finish load dataset")

    checkpoint_name = "imthanhlv/vigpt2medium"
    LOGGER.info(f"Start load model and tokenizer from checkpoint: {checkpoint_name}")
    model = load_model(checkpoint_name)
    tokenizer = load_tokenizer(checkpoint_name)
    LOGGER.info(f"Finish load model and tokenizer")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

    num_gpu = torch.cuda.device_count()
    LOGGER.info(f"Numer of GPU training: {num_gpu}")

    with open("./config/training_config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
    LOGGER.info(f"Training config: {config}")

    step_per_batch = math.ceil(len(train_dataset)/(config["batch_size"]*config["gradient_accumulation_steps"]*num_gpu))
    LOGGER.info(f"Step per batch: {step_per_batch}")

    training_args = TrainingArguments(
        output_dir='training_article',
        # group_by_length=True,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        num_train_epochs=config["training_epoch"],
        # fp16=True,
        save_steps=2500,
        eval_steps=2500,
        logging_steps=500,
        learning_rate=config["learning_rate"],
        warmup_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True,
        prediction_loss_only=True,
        metric_for_best_model='loss',
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01
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



