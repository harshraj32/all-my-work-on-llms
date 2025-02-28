# Knowledge Distillation: Llama-2 to RoBERTa on HelpSteer Dataset

This project demonstrates knowledge distillation from a large language model (Llama-2) to a smaller model (RoBERTa) for the HelpSteer2 dataset, followed by quantization for efficient deployment.

## Project Structure

```
llama+helpsteer/
├── data_preparation.py      # Prepares and processes the HelpSteer2 dataset
├── training_teacher_model.py # Trains Llama-2 as teacher using 8-bit quantization and LoRA
├── kd.py                    # Implements knowledge distillation from teacher to student
├── upload_and_load_model.py # Handles model uploading, quantization, and loading
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd knowledge-distillation+quantization/llama+helpsteer

# Install required packages
pip install -r requirements.txt
```

## Required Packages

```
transformers>=4.30.0
datasets>=2.12.0
torch>=2.0.0
scikit-learn>=1.0.2
matplotlib>=3.5.2
seaborn>=0.12.0
peft>=0.4.0
optimum[onnxruntime]>=1.8.0
huggingface_hub>=0.16.0
```

## Usage

### 1. Data Preparation

```bash
python data_preparation.py
```
This script:
- Loads the HelpSteer2 dataset
- Processes and formats the data for classification
- Creates train/validation/test splits
- Saves the processed dataset to disk

### 2. Training the Teacher Model

```bash
python training_teacher_model.py
```
This script:
- Loads the processed dataset
- Initializes Llama-2 model with 8-bit quantization
- Applies LoRA for efficient fine-tuning
- Trains the model on the HelpSteer2 dataset
- Saves the trained teacher model

### 3. Knowledge Distillation

```bash
python kd.py
```
This script:
- Loads the trained teacher model
- Initializes a smaller RoBERTa model as the student
- Performs knowledge distillation
- Saves the trained student model

### 4. Model Upload and Quantization

```bash
python upload_and_load_model.py --hf_token YOUR_HF_TOKEN --repo_name your-username/model-name
```
This script:
- Uploads the trained student model to Hugging Face Hub
- Quantizes the model using ONNX Runtime
- Provides functions to load and test the quantized model

## Model Architecture

### Teacher Model
- Base Model: Llama-2-7b
- Quantization: 8-bit
- Fine-tuning: LoRA
- Task: Multi-class classification on HelpSteer2 dataset

### Student Model
- Base Model: RoBERTa-base
- Training: Knowledge Distillation
- Quantization: ONNX Runtime
- Size: ~500MB (compared to teacher's ~7GB)

## Performance Metrics

The project tracks several metrics:
- Accuracy
- F1 Score
- Precision
- Recall

Metrics are logged during:
- Teacher model training
- Knowledge distillation
- Final evaluation of quantized student model

## Additional Notes

1. **Memory Requirements**:
   - Teacher training: ~16GB GPU RAM (with 8-bit quantization)
   - Distillation: ~8GB GPU RAM
   - Quantized student inference: ~2GB RAM

2. **Training Time**:
   - Teacher: ~4-6 hours on a single V100
   - Distillation: ~1-2 hours on a single V100
   - Quantization: ~10 minutes

3. **Model Sizes**:
   - Teacher (Llama-2): ~7GB
   - Student (RoBERTa): ~500MB
   - Quantized Student: ~125MB

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

