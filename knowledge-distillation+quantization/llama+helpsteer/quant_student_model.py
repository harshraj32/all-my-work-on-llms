import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    print("Installing bitsandbytes for quantization...")
    !pip install -q bitsandbytes>=0.39.0
    from transformers import BitsAndBytesConfig

try:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    ONNX_AVAILABLE = True
except ImportError:
    print("ONNX Runtime not available, installing...")
    !pip install -q optimum[onnxruntime]
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        ONNX_AVAILABLE = True
    except ImportError:
        ONNX_AVAILABLE = False
        print("ONNX Runtime still not available, will skip ONNX quantization")

def quantize_student_model(
    model_path="roberta-helpsteer-student",
    data_path="processed_helpsteer",
    output_dir="quantized_models"
):
    """
    Quantize the student model to make it even smaller and faster
    """
    print("Loading student model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Original model information
    original_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    original_size_mb = original_size_bytes / (1024 * 1024)
    original_params = sum(p.numel() for p in model.parameters())
    
    print(f"Original model size: {original_size_mb:.2f} MB ({original_params:,} parameters)")
    
    # Load test dataset for evaluation
    print("Loading test dataset...")
    dataset = load_from_disk(data_path)
    test_dataset = dataset["test"]
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # Tokenize test dataset if needed
    if "input_ids" not in test_dataset.column_names:
        test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "context", "multi_labels"]
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Evaluate original model for baseline
    def evaluate_model(model, dataset):
        model.eval()
        model.to(device)
        
        all_preds = []
        all_labels = []
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32
        )
        
        # Timing information
        start_time = time.time()
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                outputs = model(**batch)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
            "inference_time": inference_time,
            "samples_per_second": len(dataset) / inference_time
        }
        
        return metrics
    
    print("Evaluating original model...")
    original_metrics = evaluate_model(model, test_dataset)
    
    results = {
        "original": {
            "size_mb": original_size_mb,
            "parameters": original_params,
            "metrics": original_metrics
        }
    }
    
    print(f"Original model metrics:")
    for metric, value in original_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # 2. Dynamic Quantization (PyTorch)
    print("\nApplying PyTorch dynamic quantization...")
    
    # Quantize the model (8-bit dynamic quantization)
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8  # the target dtype for quantized weights
    )
    
    # Save quantized model
    torch_quantized_dir = os.path.join(output_dir, "dynamic_quantized")
    os.makedirs(torch_quantized_dir, exist_ok=True)
    torch.save(quantized_model.state_dict(), os.path.join(torch_quantized_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(torch_quantized_dir)
    
    # Get quantized model size
    quantized_size_bytes = sum(p.numel() * (1 if p.dtype == torch.qint8 else p.element_size()) 
                            for p in quantized_model.parameters())
    quantized_size_mb = quantized_size_bytes / (1024 * 1024)
    quantized_params = sum(p.numel() for p in quantized_model.parameters())
    
    print(f"PyTorch quantized model size: {quantized_size_mb:.2f} MB ({quantized_params:,} parameters)")
    print(f"Size reduction: {(1 - quantized_size_mb/original_size_mb) * 100:.2f}%")
    
    # Evaluate quantized model
    print("Evaluating PyTorch quantized model...")
    quantized_metrics = evaluate_model(quantized_model, test_dataset)
    
    results["pytorch_dynamic"] = {
        "size_mb": quantized_size_mb,
        "parameters": quantized_params,
        "metrics": quantized_metrics
    }
    
    print(f"PyTorch quantized model metrics:")
    for metric, value in quantized_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # 3. Hugging Face BitsAndBytes 4-bit Quantization
    print("\nApplying Hugging Face 4-bit quantization...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model with 4-bit quantization
    model_4bit = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Save config for 4-bit model
    hf_4bit_dir = os.path.join(output_dir, "4bit_quantized")
    os.makedirs(hf_4bit_dir, exist_ok=True)
    model_4bit.config.save_pretrained(hf_4bit_dir)
    tokenizer.save_pretrained(hf_4bit_dir)
    
    # Estimate size (rough approximation since we can't directly measure 4-bit models)
    estimated_4bit_size_mb = original_size_mb * (4/32)  # 4-bit vs 32-bit
    
    print(f"Estimated HF 4-bit model size: {estimated_4bit_size_mb:.2f} MB")
    print(f"Estimated size reduction: {(1 - estimated_4bit_size_mb/original_size_mb) * 100:.2f}%")
    
    # Evaluate 4-bit model
    print("Evaluating HF 4-bit quantized model...")
    model_4bit_metrics = evaluate_model(model_4bit, test_dataset)
    
    results["hf_4bit"] = {
        "size_mb": estimated_4bit_size_mb,
        "parameters": original_params,  # same number of parameters, just quantized
        "metrics": model_4bit_metrics
    }
    
    print(f"HF 4-bit quantized model metrics:")
    for metric, value in model_4bit_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # 4. ONNX Quantization (optional)
    if ONNX_AVAILABLE:
        print("\nConverting to ONNX and applying ONNX quantization...")
        
        # ONNX export directory
        onnx_dir = os.path.join(output_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        
        # Export to ONNX
        from optimum.onnxruntime import ORTModelForSequenceClassification
        
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_path, export=True
        )
        
        # Save ONNX model
        ort_model.save_pretrained(onnx_dir)
        tokenizer.save_pretrained(onnx_dir)
        
        # Quantize ONNX model
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        
        # Apply dynamic quantization
        quantizer.quantize(
            quantization_config=qconfig,
            save_dir=os.path.join(output_dir, "onnx_quantized")
        )
        
        print("ONNX quantization complete")
    
    # Create comparison visualizations
    methods = list(results.keys())
    sizes = [results[method]["size_mb"] for method in methods]
    accuracies = [results[method]["metrics"]["accuracy"] for method in methods]
    speeds = [results[method]["metrics"]["samples_per_second"] for method in methods]
    
    # Size comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(methods, sizes)
    plt.title('Model Size Comparison')
    plt.ylabel('Size (MB)')
    plt.xticks(rotation=45)
    
    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.bar(methods, accuracies)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparisons.png"))
    
    # Speed comparison
    plt.figure(figsize=(10, 6))
    plt.bar(methods, speeds)
    plt.title('Speed Comparison')
    plt.ylabel('Samples per second')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speed_comparison.png"))
    
    # Save results to JSON
    with open(os.path.join(output_dir, "quantization_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nQuantization complete. Results saved to {output_dir}")
    return results

if __name__ == "__main__":
    quantize_student_model()