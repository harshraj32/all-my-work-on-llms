import json
import os

import torch
from huggingface_hub import HfApi, login
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def upload_model_to_hub(
    model_path="roberta-helpsteer-student",
    repo_name="your-username/roberta-helpsteer-student",
    token=None,
):
    """
    Upload the trained student model to Hugging Face Hub
    """
    if token is None:
        print("Please provide your Hugging Face token!")
        return

    # Login to Hugging Face
    login(token)
    api = HfApi()

    print(f"Uploading model from {model_path} to {repo_name}")

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_name, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload model files
    api.upload_folder(folder_path=model_path, repo_id=repo_name, repo_type="model")

    print("Model uploaded successfully!")


def quantize_and_export_model(
    model_path="roberta-helpsteer-student",
    output_dir="roberta-helpsteer-student-quantized",
    optimization_level=99,
):
    """
    Quantize the model using ONNX Runtime and export it
    """
    print("Loading model for quantization...")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create quantizer
    quantizer = ORTQuantizer.from_pretrained(model_path)

    # Define quantization configuration
    qconfig = AutoQuantizationConfig.arm64(
        is_static=True, per_channel=False, optimization_level=optimization_level
    )

    # Quantize the model
    print("Quantizing model...")
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")

    return output_dir


def load_quantized_model(
    model_path="roberta-helpsteer-student-quantized",
    sample_text="Hello, how can I help you today?",
):
    """
    Load and test the quantized model
    """
    print("Loading quantized model...")

    # Load model and tokenizer
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        provider="CPUExecutionProvider",  # or "CUDAExecutionProvider" for GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Test inference
    print("\nTesting inference with sample text...")
    result = classifier(sample_text)
    print(f"Input text: {sample_text}")
    print(f"Classification result: {result}")

    return classifier


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload, quantize, and load the student model"
    )
    parser.add_argument("--hf_token", type=str, help="Hugging Face API token")
    parser.add_argument(
        "--repo_name", type=str, default="your-username/roberta-helpsteer-student"
    )
    parser.add_argument("--model_path", type=str, default="roberta-helpsteer-student")
    parser.add_argument(
        "--quantized_path", type=str, default="roberta-helpsteer-student-quantized"
    )
    args = parser.parse_args()

    # Upload model to Hub
    if args.hf_token:
        upload_model_to_hub(args.model_path, args.repo_name, args.hf_token)

    # Quantize model
    quantized_path = quantize_and_export_model(args.model_path, args.quantized_path)

    # Load and test quantized model
    classifier = load_quantized_model(quantized_path)
