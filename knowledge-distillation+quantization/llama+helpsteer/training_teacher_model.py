import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import json

# Set seed for reproducibility
set_seed(42)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_llama_teacher(
    data_path="processed_helpsteer",
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="llama-helpsteer-teacher"
):
    """
    Train Llama-2 model as a teacher on HelpSteer2 dataset
    using 8-bit quantization and LoRA for memory efficiency.
    """
    print("Loading processed dataset...")
    dataset = load_from_disk(data_path)
    
    # Load metadata
    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    id2label = metadata["id2label"]
    label2id = metadata["label2id"]
    num_labels = metadata["num_labels"]
    
    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    print(f"Classification task with {num_labels} labels")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Define tokenization function
    def tokenize_function(examples):
        # For Llama models, we format inputs with clear instruction format
        texts = [
            f"<s>[INST] Classify the following response text. Context: {context}\n\nText: {text} [/INST]"
            for text, context in zip(examples["text"], examples["context"])
        ]
        
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return tokenized
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "context", "multi_labels"]
    )
    
    print("Dataset tokenized.")
    print(f"Train dataset size: {len(tokenized_datasets['train'])}")
    print(f"Validation dataset size: {len(tokenized_datasets['validation'])}")
    
    # Define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Prepare evaluation metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted")
        }
    
    # Load model in 8-bit mode
    print("Loading Llama-2 model in 8-bit mode...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Prepare the model for PEFT/LoRA fine-tuning
    print("Preparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        max_grad_norm=0.3,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=10,
        fp16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    # Save model and tokenizer
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save test results
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f)
    
    print(f"Training complete. Model saved to {output_dir}")
    return model, tokenizer, test_results

if __name__ == "__main__":
    train_llama_teacher()