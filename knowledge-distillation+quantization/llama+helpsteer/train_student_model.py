import argparse
import gc
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

# Set seed for reproducibility
set_seed(42)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DistillationTrainer:
    """
    Custom trainer for knowledge distillation from a teacher model to a student model
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        tokenizer,
        teacher_tokenizer=None,
        args=None,
        alpha=0.5,
        temperature=2.0,
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer if teacher_tokenizer else tokenizer
        self.args = args
        self.alpha = alpha  # Weight for distillation loss vs hard-label loss
        self.temperature = temperature  # Temperature for softening distributions

        # Define optimization parameters
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Set models to proper modes
        self.teacher_model.eval()  # Teacher is always in evaluation mode

        # DataLoader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        self.eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        # Scheduler
        num_training_steps = args.num_train_epochs * len(self.train_dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.learning_rate,
            total_steps=num_training_steps,
        )

        # For tracking metrics
        self.best_eval_metric = 0
        self.best_model_state = None

    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute the combined distillation loss:
        - KL divergence between softened teacher and student distributions
        - Cross-entropy loss with hard labels
        """
        # Distillation loss: KL divergence between softened distributions
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (
            self.temperature**2
        )

        # Standard cross-entropy loss with hard labels
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return loss, distillation_loss, hard_loss

    def evaluation_step(self):
        """Evaluate the student model on the evaluation dataset"""
        self.student_model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                outputs = self.student_model(**batch)
                logits = outputs.logits

                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")

        avg_loss = total_loss / len(self.eval_dataloader)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        # Save best model if current model is better
        if accuracy > self.best_eval_metric:
            self.best_eval_metric = accuracy
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.student_model.state_dict().items()
            }

        return metrics

    def train(self):
        """Train the student model with knowledge distillation from the teacher"""
        num_epochs = self.args.num_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        log_every = max(1, steps_per_epoch // 10)  # Log about 10 times per epoch

        print(
            f"Starting training for {num_epochs} epochs with {steps_per_epoch} steps per epoch"
        )
        global_step = 0

        for epoch in range(num_epochs):
            self.student_model.train()
            epoch_loss = 0
            epoch_distill_loss = 0
            epoch_hard_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                # Get student predictions
                outputs = self.student_model(**batch)
                student_logits = outputs.logits

                # Get teacher predictions (without gradient tracking)
                with torch.no_grad():
                    # Handle conversion between different tokenizers if needed
                    if self.teacher_tokenizer != self.tokenizer:
                        # This is a placeholder - in a real scenario, you'd need to handle
                        # conversion between different tokenizers more carefully
                        teacher_batch = batch
                    else:
                        teacher_batch = batch

                    teacher_outputs = self.teacher_model(**teacher_batch)
                    teacher_logits = teacher_outputs.logits

                # Compute distillation loss
                loss, distill_loss, hard_loss = self.compute_distillation_loss(
                    student_logits, teacher_logits, labels
                )

                # Backward pass and optimization
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), max_norm=1.0
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_distill_loss += distill_loss.item()
                epoch_hard_loss += hard_loss.item()
                global_step += 1

                # Log progress
                if step % log_every == 0 or step == steps_per_epoch - 1:
                    print(
                        f"Epoch: {epoch+1}/{num_epochs} | "
                        f"Step: {step+1}/{steps_per_epoch} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Distill Loss: {distill_loss.item():.4f} | "
                        f"Hard Loss: {hard_loss.item():.4f}"
                    )

            # Calculate average losses for the epoch
            avg_loss = epoch_loss / steps_per_epoch
            avg_distill_loss = epoch_distill_loss / steps_per_epoch
            avg_hard_loss = epoch_hard_loss / steps_per_epoch

            print(
                f"Epoch {epoch+1}/{num_epochs} completed | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Avg Distill Loss: {avg_distill_loss:.4f} | "
                f"Avg Hard Loss: {avg_hard_loss:.4f}"
            )

            # Evaluate after each epoch
            eval_metrics = self.evaluation_step()

            print(
                f"Eval Metrics - "
                f"Loss: {eval_metrics['loss']:.4f} | "
                f"Accuracy: {eval_metrics['accuracy']:.4f} | "
                f"F1: {eval_metrics['f1']:.4f} | "
                f"Precision: {eval_metrics['precision']:.4f} | "
                f"Recall: {eval_metrics['recall']:.4f}"
            )

        # Load the best model state
        print(
            f"Training complete. Loading best model with accuracy: {self.best_eval_metric:.4f}"
        )
        self.student_model.load_state_dict(self.best_model_state)
        return self.student_model


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


def distill_llama3_to_llama2(
    teacher_path="llama3-helpsteer-teacher",
    data_path="processed_helpsteer/fold_1",
    output_dir="llama2-helpsteer-student",
    student_model_id="meta-llama/Llama-2-7b-hf",
    benchmark_before=True,
    benchmark_after=True,
    alpha=0.7,
    temperature=4.0,
    num_epochs=3,
):
    """
    Knowledge distillation from Llama-3 teacher to Llama-2 student
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

    # Load teacher model and tokenizer
    print("Loading Llama-3 teacher model...")

    # Two approaches depending on whether we saved a PEFT model or full model
    try:
        # Try loading as PEFT model first
        teacher_config = PeftModel.from_pretrained(teacher_path)
        base_model_id = teacher_config.base_model_name_or_path

        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id, num_labels=num_labels, load_in_8bit=True, device_map="auto"
        )
        teacher_model = PeftModel.from_pretrained(teacher_model, teacher_path)

    except:
        # Fall back to loading as a regular model
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            teacher_path, load_in_8bit=True, device_map="auto"
        )

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)

    # Load student model and tokenizer
    print("Initializing Llama-2 student model...")

    # Benchmark student model before training if requested
    if benchmark_before:
        print("\nBenchmarking student model before training...")
        # Create sample texts from the dataset for benchmarking
        sample_texts = []
        for i in range(min(5, len(dataset["validation"]))):
            sample_texts.append(
                f"Classify the following response based on helpfulness. Context: {dataset['validation'][i]['context']}\n\nText: {dataset['validation'][i]['text']}"
            )
        benchmark_model(student_model_id, None, sample_texts)

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)

    # Add padding token if needed
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # Load student model in 8-bit mode
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare the student model for PEFT/LoRA fine-tuning
    print("Preparing student model for LoRA fine-tuning...")
    student_model = prepare_model_for_kbit_training(student_model)

    # Configure LoRA for student
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

    # Apply LoRA to the student model
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # Define tokenization function for student model
    def tokenize_function(examples):
        # For Llama models, we use the same format for both teacher and student
        texts = [
            f"<s>[INST] Classify the following response based on helpfulness. Context: {context}\n\nText: {text} [/INST]"
            for text, context in zip(examples["text"], examples["context"])
        ]

        return student_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    # Tokenize datasets
    print("Tokenizing datasets for the student model...")
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,
    )

    # Initialize distillation trainer
    print("Starting knowledge distillation...")
    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        args=training_args,
        alpha=alpha,  # Weight on distillation loss
        temperature=temperature,  # Temperature for softer distributions
    )

    # Train with knowledge distillation
    print("Training student model with knowledge distillation...")
    start_time = time.time()
    student_model = distillation_trainer.train()
    end_time = time.time()

    print(f"Distillation training completed in {end_time - start_time:.2f} seconds")

    # Evaluate student model on test set
    print("\nEvaluating student model on test set...")
    test_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["test"],
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=student_tokenizer),
    )

    student_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = student_model(**batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    test_precision = precision_score(all_labels, all_preds, average="weighted")
    test_recall = recall_score(all_labels, all_preds, average="weighted")

    test_results = {
        "accuracy": test_accuracy,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
    }

    print("Student model test results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")

    # Compare model sizes
    teacher_size_mb = sum(p.numel() * 1 for p in teacher_model.parameters()) / (
        1024 * 1024 * 8
    )  # 8-bit model
    student_size_mb = sum(p.numel() * 1 for p in student_model.parameters()) / (
        1024 * 1024 * 8
    )  # 8-bit model

    print(f"\nModel size comparison:")
    print(f"  Teacher model (8-bit): {teacher_size_mb:.2f} MB")
    print(f"  Student model (8-bit): {student_size_mb:.2f} MB")
    print(f"  Size reduction: {(1 - student_size_mb/teacher_size_mb) * 100:.2f}%")

    # Save student model and tokenizer
    print(f"\nSaving student model to {output_dir}")
    student_model.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)

    # Save test results
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # Benchmark student model after training if requested
    if benchmark_after:
        print("\nBenchmarking student model after training...")
        # Create sample texts from the dataset for benchmarking
        sample_texts = []
        for i in range(min(5, len(dataset["test"]))):
            sample_texts.append(
                f"Classify the following response based on helpfulness. Context: {dataset['test'][i]['context']}\n\nText: {dataset['test'][i]['text']}"
            )
        benchmark_model(output_dir, student_tokenizer, sample_texts)

    return student_model, student_tokenizer, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distill Llama-3 teacher to Llama-2 student"
    )
    parser.add_argument("--fold", type=int, default=1, help="Which fold to use (1-10)")
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="llama3-helpsteer-teacher",
        help="Path to teacher model",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Student model ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llama2-helpsteer-student",
        help="Output directory",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Weight for distillation loss (0-1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Temperature for softening distributions",
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

    distill_llama3_to_llama2(
        teacher_path=args.teacher_path,
        data_path=data_path,
        output_dir=args.output_dir,
        student_model_id=args.student_model,
        benchmark_before=not args.no_benchmark_before,
        benchmark_after=not args.no_benchmark_after,
        alpha=args.alpha,
        temperature=args.temperature,
        num_epochs=args.epochs,
    )
