import argparse
import gc
import json
import os
import time

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Set seed for reproducibility
set_seed(42)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def benchmark_model(model_id, tokenizer=None, sample_texts=None):
    """
    Benchmark the model using sample texts
    """
    print(f"\nBenchmarking model: {model_id}")

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return {"error": str(e)}

    # Default sample texts if none provided
    if sample_texts is None:
        sample_texts = [
            "I need help with my homework on calculus.",
            "Can you explain how photosynthesis works?",
            "What are the main causes of climate change?",
            "How do I bake a chocolate cake?",
            "What's the best way to learn a new language?",
        ]

    # Try to load the model
    try:
        # For benchmarking, we'll use a pipeline if possible
        from transformers import pipeline

        # For classification models
        if os.path.exists(model_id):
            # Local model
            with open(os.path.join(model_id, "config.json"), "r") as f:
                config = json.load(f)
            if "id2label" in config:
                # It's a classification model
                classifier = pipeline(
                    "text-classification", model=model_id, tokenizer=tokenizer
                )
                results = {"classification_results": {}}

                for i, text in enumerate(sample_texts):
                    try:
                        result = classifier(text)
                        results["classification_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "prediction": result,
                        }
                    except Exception as e:
                        results["classification_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "error": str(e),
                        }
            else:
                # Assume it's a text generation model
                generator = pipeline(
                    "text-generation", model=model_id, tokenizer=tokenizer
                )
                results = {"generation_results": {}}

                for i, text in enumerate(sample_texts):
                    try:
                        prompt = f"<s>[INST] {text} [/INST]"
                        result = generator(
                            prompt, max_length=100, num_return_sequences=1
                        )
                        results["generation_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "generation": result,
                        }
                    except Exception as e:
                        results["generation_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "error": str(e),
                        }
        else:
            # HuggingFace model
            try:
                # Try classification first
                classifier = pipeline(
                    "text-classification", model=model_id, tokenizer=tokenizer
                )
                results = {"classification_results": {}}

                for i, text in enumerate(sample_texts):
                    try:
                        result = classifier(text)
                        results["classification_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "prediction": result,
                        }
                    except Exception as e:
                        results["classification_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "error": str(e),
                        }
            except:
                # Fall back to text generation
                generator = pipeline(
                    "text-generation", model=model_id, tokenizer=tokenizer
                )
                results = {"generation_results": {}}

                for i, text in enumerate(sample_texts):
                    try:
                        prompt = f"<s>[INST] {text} [/INST]"
                        result = generator(
                            prompt, max_length=100, num_return_sequences=1
                        )
                        results["generation_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "generation": result,
                        }
                    except Exception as e:
                        results["generation_results"][f"sample_{i+1}"] = {
                            "text": text,
                            "error": str(e),
                        }
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        results = {"error": str(e)}

    # Save results to file
    benchmark_dir = "benchmark_results"
    os.makedirs(benchmark_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(
        benchmark_dir, f"{model_id.replace('/', '_')}_{timestamp}.json"
    )

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark results saved to {result_file}")
    return results


def train_llama_teacher(
    data_path="processed_helpsteer/fold_1",  # Default to first fold
    model_id="meta-llama/Llama-3-8B",  # Using Llama-3 as teacher
    output_dir="llama3-helpsteer-teacher",
    benchmark_before=True,
    benchmark_after=True,
    num_epochs=3,
):
    """
    Train Llama-3 model as a teacher on HelpSteer2 dataset
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
    fold_info = metadata.get("fold_info", {"current_fold": 1, "total_folds": 10})

    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    print(f"Classification task with {num_labels} labels")
    print(f"Using fold {fold_info['current_fold']} of {fold_info['total_folds']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Benchmark model before training if requested
    if benchmark_before:
        print("\nBenchmarking model before training...")
        # Create sample texts from the dataset for benchmarking
        sample_texts = []
        for i in range(min(5, len(dataset["validation"]))):
            sample_texts.append(
                f"Classify the following response based on helpfulness. Context: {dataset['validation'][i]['context']}\n\nText: {dataset['validation'][i]['text']}"
            )
        benchmark_model(model_id, tokenizer, sample_texts)

    # Define tokenization function
    def tokenize_function(examples):
        # For Llama models, we format inputs with clear instruction format
        texts = [
            f"<s>[INST] Classify the following response based on helpfulness. Context: {context}\n\nText: {text} [/INST]"
            for text, context in zip(examples["text"], examples["context"])
        ]

        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return tokenized

    # Tokenize datasets
    print("Tokenizing datasets...")
    # Get all columns that should be removed (all except 'labels')
    columns_to_remove = [
        col for col in dataset["train"].column_names if col != "labels"
    ]

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=columns_to_remove
    )

    print("Dataset tokenized.")
    print(f"Train dataset size: {len(tokenized_datasets['train'])}")
    print(f"Validation dataset size: {len(tokenized_datasets['validation'])}")
    print(f"Test dataset size: {len(tokenized_datasets['test'])}")

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
            "recall": recall_score(labels, predictions, average="weighted"),
        }

    # Load model in 8-bit mode
    print(f"Loading {model_id} model in 8-bit mode...")
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
        num_train_epochs=num_epochs,
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

    # Benchmark model after training if requested
    if benchmark_after:
        print("\nBenchmarking model after training...")
        # Create sample texts from the dataset for benchmarking
        sample_texts = []
        for i in range(min(5, len(dataset["test"]))):
            sample_texts.append(
                f"Classify the following response based on helpfulness. Context: {dataset['test'][i]['context']}\n\nText: {dataset['test'][i]['text']}"
            )
        benchmark_model(output_dir, tokenizer, sample_texts)

    print(f"Training complete. Model saved to {output_dir}")
    return model, tokenizer, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Llama-3 teacher model on HelpSteer2 dataset"
    )
    parser.add_argument("--fold", type=int, default=1, help="Which fold to use (1-10)")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3-8B", help="Teacher model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llama3-helpsteer-teacher",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--no_benchmark_before",
        action="store_true",
        help="Skip benchmarking before training",
    )
    parser.add_argument(
        "--no_benchmark_after",
        action="store_true",
        help="Skip benchmarking after training",
    )

    args = parser.parse_args()

    data_path = f"processed_helpsteer/fold_{args.fold}"

    train_llama_teacher(
        data_path=data_path,
        model_id=args.model,
        output_dir=args.output_dir,
        benchmark_before=not args.no_benchmark_before,
        benchmark_after=not args.no_benchmark_after,
        num_epochs=args.epochs,
    )
