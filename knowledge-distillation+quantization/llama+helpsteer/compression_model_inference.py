import torch
import json
import time
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path, quantization_type=None):
    """
    Load a model with the specified quantization type
    
    Args:
        model_path: Path to the model directory
        quantization_type: None, "dynamic", "4bit", or "onnx"
    
    Returns:
        model, tokenizer
    """
    if quantization_type == "4bit":
        from transformers import BitsAndBytesConfig
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
    elif quantization_type == "onnx":
        from optimum.onnxruntime import ORTModelForSequenceClassification
        
        model = ORTModelForSequenceClassification.from_pretrained(model_path)
        
    elif quantization_type == "dynamic":
        # For dynamic quantization, we load the base model and then apply quantization
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
    else:
        # No quantization, just load the standard model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(model, tokenizer, text, context=None, device="cpu"):
    """
    Make a prediction using the model
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        text: The text to classify
        context: Optional context to include in the input
        device: The device to run inference on
    
    Returns:
        prediction dict with label, confidence, and inference time
    """
    # Move model to device if not already
    if hasattr(model, "to") and not isinstance(model, torch.nn.DataParallel):
        model.to(device)
    
    # Format input based on whether context is provided
    if context:
        input_text = f"Context: {context}\n\nText: {text}"
    else:
        input_text = text
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference with timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    inference_time = time.time() - start_time
    
    # Get prediction
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[predicted_class].item()
    
    # Get label name
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        label = model.config.id2label[predicted_class]
    else:
        label = f"Class {predicted_class}"
    
    return {
        "label": label,
        "confidence": confidence,
        "inference_time": inference_time,
        "probabilities": {
            model.config.id2label[i]: prob.item() 
            for i, prob in enumerate(probabilities)
        } if hasattr(model, "config") and hasattr(model.config, "id2label") else None
    }

def main():
    parser = argparse.ArgumentParser(description="Run inference with a compressed model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "dynamic", "4bit", "onnx"], 
                        help="Quantization type to use")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--context", type=str, default=None, help="Optional context for the text")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path} with {args.quantization} quantization...")
    model, tokenizer = load_model(args.model_path, args.quantization)
    
    print(f"Running inference on device: {args.device}")
    result = predict(model, tokenizer, args.text, args.context, args.device)
    
    print("\nPrediction Results:")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Inference Time: {result['inference_time'] * 1000:.2f} ms")
    
    if result["probabilities"]:
        print("\nProbabilities for all classes:")
        for label, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()