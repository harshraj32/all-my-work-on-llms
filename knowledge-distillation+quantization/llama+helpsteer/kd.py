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
from peft import PeftConfig, PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    RobertaTokenizer,
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


def distill_llama_to_roberta(
    teacher_path="llama-helpsteer-teacher",
    data_path="processed_helpsteer",
    output_dir="roberta-helpsteer-student",
):
    """
    Knowledge distillation from Llama-2 (8-bit) teacher to RoBERTa student
    """
    print("Loading processed dataset...")
    dataset = load_from_disk(data_path)

    # Load metadata
    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    id2label = metadata["id2label"]
    label2id = metadata["label2id"]
    num_labels = metadata["num_labels"]

    # Load teacher model and tokenizer
    print("Loading Llama-2 teacher model...")

    # Two approaches depending on whether we saved a PEFT model or full model
    try:
        # Try loading as PEFT model first
        teacher_config = PeftConfig.from_pretrained(teacher_path)
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

    # Load student model and tokenizer (using smaller RoBERTa model)
    print("Initializing RoBERTa student model...")
    student_model_name = "roberta-base"  # Much smaller than Llama-2
    student_tokenizer = RobertaTokenizer.from_pretrained(student_model_name)
    student_model = RobertaForSequenceClassification.from_pretrained(
        student_model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # Add padding token if needed
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # Define tokenization function for student model
    def tokenize_function(examples):
        # For RoBERTa, we use a simpler format without specific instruction template
        return student_tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    # Tokenize datasets
    print("Tokenizing datasets for the student model...")
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        args=training_args,
        alpha=0.5,  # Balance between distillation and hard label loss
        temperature=2.0,  # Temperature for softening probability distributions
    )

    # Train and get the best model
    print("Starting knowledge distillation training...")
    best_student_model = trainer.train()

    # Save the final student model
    print(f"Saving student model to {output_dir}")
    best_student_model.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)

    # Clean up to free memory
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    return best_student_model


if __name__ == "__main__":
    # Example usage
    distill_llama_to_roberta(
        teacher_path="llama-helpsteer-teacher",
        data_path="processed_helpsteer",
        output_dir="roberta-helpsteer-student",
    )
