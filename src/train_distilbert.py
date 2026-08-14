import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
from .config import config
from .data_loader import load_distilbert_data

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_distilbert():
    print("=" * 60)
    print("Training DistilBERT + LoRA")
    print("=" * 60)
    
    data = load_distilbert_data()
    tokenized_dataset = data["tokenized_dataset"]
    tokenizer = data["tokenizer"]
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.distilbert_model, num_labels=config.num_labels)
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    training_args = TrainingArguments(
        output_dir=config.distilbert_path,
        num_train_epochs=config.num_epochs_distilbert,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        learning_rate=config.learning_rate,
    )
    
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name="distilbert_lora"):
        mlflow.log_params({
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "trainable_params": trainable_params
        })
        
        # FIXED: Removed tokenizer parameter
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics,
        )
        trainer.train()
        
        eval_results = trainer.evaluate()
        print(f"Results: {eval_results}")
        for key, value in eval_results.items():
            mlflow.log_metric(key, value)
        
        model.save_pretrained(config.distilbert_path)
        tokenizer.save_pretrained(config.distilbert_path)
        print(f"Model saved to: {config.distilbert_path}")
        return trainer, model

if __name__ == "__main__":
    train_distilbert()