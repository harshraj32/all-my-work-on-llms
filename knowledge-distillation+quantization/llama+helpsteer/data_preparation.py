import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def prepare_helpsteer_data(n_splits=10, fold_idx=0, seed=42):
    """
    Load and prepare the HelpSteer2 dataset for classification with k-fold cross-validation.

    Args:
        n_splits (int): Number of folds for cross-validation (default: 10)
        fold_idx (int): Which fold to use as validation (0 to n_splits-1)
        seed (int): Random seed for reproducibility

    Returns:
        processed_dataset: DatasetDict with train, validation, and test splits
        metadata: Dictionary containing label mappings and other metadata
    """
    print("Loading HelpSteer2 dataset...")

    # Load the dataset
    dataset = load_dataset("nvidia/HelpSteer2", trust_remote_code=True)

    print(
        f"Dataset loaded with {len(dataset['train'])} training and {len(dataset['validation'])} validation examples"
    )

    # Print available columns
    print("\nAvailable columns:", dataset["train"].column_names)

    # We'll create labels based on the helpfulness score
    # This converts the regression problem into a classification task
    def create_helpfulness_label(score):
        if score < 0.33:
            return "low_helpfulness"
        elif score < 0.66:
            return "medium_helpfulness"
        else:
            return "high_helpfulness"

    # Create label encoder
    label_encoder = LabelEncoder()

    def process_examples(examples):
        # Create labels from helpfulness scores
        text_labels = [
            create_helpfulness_label(score) for score in examples["helpfulness"]
        ]

        # Fit label encoder if not fitted
        if not hasattr(label_encoder, "classes_"):
            label_encoder.fit(
                ["low_helpfulness", "medium_helpfulness", "high_helpfulness"]
            )

        # Convert text labels to numeric
        numeric_labels = label_encoder.transform(text_labels)

        return {
            "text": examples["response"],
            "context": examples["prompt"],
            "labels": numeric_labels,
            "helpfulness": examples["helpfulness"],
            "correctness": examples["correctness"],
            "coherence": examples["coherence"],
            "complexity": examples["complexity"],
            "verbosity": examples["verbosity"],
        }

    # Process both train and validation sets
    processed_train = process_examples(dataset["train"])
    processed_validation = process_examples(dataset["validation"])

    # Create datasets
    train_dataset = Dataset.from_dict(processed_train)
    holdout_dataset = Dataset.from_dict(
        processed_validation
    )  # This will be our test set

    # Convert train dataset to numpy for k-fold splitting
    train_data = {
        "text": train_dataset["text"],
        "context": train_dataset["context"],
        "labels": train_dataset["labels"],
        "helpfulness": train_dataset["helpfulness"],
        "correctness": train_dataset["correctness"],
        "coherence": train_dataset["coherence"],
        "complexity": train_dataset["complexity"],
        "verbosity": train_dataset["verbosity"],
    }

    # Initialize k-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Get the indices for the specified fold
    train_indices = []
    val_indices = []

    for i, (train_idx, val_idx) in enumerate(kf.split(train_data["text"])):
        if i == fold_idx:
            train_indices = train_idx
            val_indices = val_idx
            break

    # Create train and validation splits based on the fold
    fold_train_dataset = Dataset.from_dict(
        {key: [train_data[key][i] for i in train_indices] for key in train_data.keys()}
    )

    fold_val_dataset = Dataset.from_dict(
        {key: [train_data[key][i] for i in val_indices] for key in train_data.keys()}
    )

    # Combine into a DatasetDict
    processed_dataset = DatasetDict(
        {
            "train": fold_train_dataset,
            "validation": fold_val_dataset,
            "test": holdout_dataset,
        }
    )

    print(f"\nFinal dataset sizes:")
    print(f"  Training: {len(fold_train_dataset)} (Fold {fold_idx + 1} of {n_splits})")
    print(f"  Validation: {len(fold_val_dataset)}")
    print(f"  Test (holdout): {len(holdout_dataset)}")

    # Create label mappings
    id2label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    label2id = {label: idx for idx, label in id2label.items()}

    # Visualize distribution of labels in the training set
    train_label_counts = {}
    for label_id in fold_train_dataset["labels"]:
        label = id2label[label_id]
        train_label_counts[label] = train_label_counts.get(label, 0) + 1

    plt.figure(figsize=(12, 6))
    plt.bar(train_label_counts.keys(), train_label_counts.values())
    plt.title(
        f"Distribution of Helpfulness Labels in Training Set (Fold {fold_idx + 1})"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs("processed_helpsteer", exist_ok=True)
    plt.savefig(
        os.path.join(
            "processed_helpsteer", f"label_distribution_fold_{fold_idx + 1}.png"
        )
    )

    # Save metadata
    metadata = {
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(label_encoder.classes_),
        "fold_info": {
            "current_fold": fold_idx + 1,
            "total_folds": n_splits,
            "seed": seed,
        },
        "metrics": [
            "helpfulness",
            "correctness",
            "coherence",
            "complexity",
            "verbosity",
        ],
    }

    return processed_dataset, metadata


if __name__ == "__main__":
    # Process dataset for each fold
    for fold_idx in range(10):
        print(f"\nProcessing fold {fold_idx + 1}/10...")
        dataset, metadata = prepare_helpsteer_data(n_splits=10, fold_idx=fold_idx)

        # Save processed dataset for this fold
        output_dir = f"processed_helpsteer/fold_{fold_idx + 1}"
        dataset.save_to_disk(output_dir)

        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved fold {fold_idx + 1} to {output_dir}")

        # Print label mapping for reference
        if fold_idx == 0:  # Only print once
            print("\nLabel mapping:")
            for label, idx in metadata["label2id"].items():
                print(f"  {label}: {idx}")
            print("\nAvailable metrics:", metadata["metrics"])
