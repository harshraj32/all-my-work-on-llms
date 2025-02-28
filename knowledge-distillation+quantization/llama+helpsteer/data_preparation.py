import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_helpsteer_data():
    """
    Load and prepare the HelpSteer2 dataset for classification.
    Returns processed dataset ready for model training.
    """
    print("Loading HelpSteer2 dataset...")
    
    # Load the dataset
    dataset = load_dataset("nvidia/HelpSteer2", trust_remote_code=True)
    
    print(f"Dataset loaded with {len(dataset['train'])} training and {len(dataset['validation'])} validation examples")
    
    # Inspect sample data
    print("\nSample data:")
    for key, value in dataset['train'][0].items():
        print(f"{key}: {value}")
    
    # The dataset contains conversations with 'resp_labels' as the target class
    # We need to convert the multi-label classification into a format suitable for training
    
    # Get unique labels
    all_labels = set()
    for split in ['train', 'validation']:
        for labels in dataset[split]['resp_labels']:
            all_labels.update(labels)
    
    label_list = sorted(list(all_labels))
    print(f"\nFound {len(label_list)} unique labels: {label_list}")
    
    # Create a label mapping
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # Function to encode multi-hot labels (for multi-label classification)
    def encode_labels(examples):
        mlb = MultiLabelBinarizer(classes=label_list)
        encoded_labels = mlb.fit_transform([examples['resp_labels']])
        return encoded_labels[0]
    
    # Process the dataset for single-label classification (take the first label)
    # This is a simplification - in reality, you might want to use multi-label classification
    def process_examples(examples):
        result = {
            'text': examples['response'],  # The text to classify
            'context': examples['context'],  # The context (we'll use this in prompting)
            'multi_labels': [encode_labels({'resp_labels': labels}) for labels in examples['resp_labels']],
            'labels': [label2id[labels[0]] if labels else 0 for labels in examples['resp_labels']]  # Take first label
        }
        return result
    
    # Process the dataset
    processed_train = process_examples(dataset['train'])
    processed_validation = process_examples(dataset['validation'])
    
    # Create new datasets with processed data
    train_dataset = Dataset.from_dict(processed_train)
    validation_dataset = Dataset.from_dict(processed_validation)
    
    # Create a test set from a portion of the validation set
    val_len = len(validation_dataset)
    test_size = val_len // 3  # Use 1/3 of validation as test set
    
    # Split validation into validation and test
    validation_dataset, test_dataset = validation_dataset.train_test_split(
        test_size=test_size, seed=42
    ).values()
    
    print(f"\nFinal dataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(validation_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Combine into a DatasetDict
    processed_dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })
    
    # Visualize distribution of labels
    train_label_counts = {}
    for label_id in processed_train['labels']:
        label = id2label[label_id]
        train_label_counts[label] = train_label_counts.get(label, 0) + 1
    
    plt.figure(figsize=(12, 6))
    plt.bar(train_label_counts.keys(), train_label_counts.values())
    plt.title('Distribution of Primary Labels in Training Set')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    
    # Save important metadata for model training
    metadata = {
        'label2id': label2id,
        'id2label': id2label,
        'num_labels': len(label_list)
    }
    
    return processed_dataset, metadata

if __name__ == "__main__":
    # Load and process the dataset
    dataset, metadata = prepare_helpsteer_data()
    
    # Save processed dataset for future use
    dataset.save_to_disk("processed_helpsteer")
    
    # Print metadata for reference
    print("\nLabel mapping:")
    for label, idx in metadata['label2id'].items():
        print(f"  {label}: {idx}")