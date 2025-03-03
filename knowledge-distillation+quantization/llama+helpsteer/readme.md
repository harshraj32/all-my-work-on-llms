# Knowledge Distillation: Llama-3 to Llama-2 on HelpSteer Dataset

This project demonstrates knowledge distillation from a larger language model (Llama-3) to a smaller model (Llama-2) for the HelpSteer2 dataset, followed by quantization for efficient deployment.

## Project Structure

```
llama+helpsteer/
├── data_preparation.py         # Prepares and processes the HelpSteer2 dataset with k-fold CV
├── training_teacher_model.py   # Trains Llama-3 as teacher using 8-bit quantization and LoRA
├── train_student_model.py      # Implements knowledge distillation from Llama-3 to Llama-2
├── upload_and_load_model.py    # Handles model uploading, quantization, and loading
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
numpy>=1.24.0
pandas>=1.5.0
tqdm>=4.65.0
tensorboard>=2.13.0
mteb>=1.0.0
```

## Usage

### 1. Data Preparation with K-Fold Cross-Validation

```bash
python data_preparation.py
```
This script:
- Loads the HelpSteer2 dataset
- Creates a classification task based on helpfulness scores
- Implements 10-fold cross-validation on the training data
- Uses the original validation set as the holdout test set
- Saves each fold separately with its metadata

### 2. Training the Llama-3 Teacher Model

```bash
python training_teacher_model.py --fold 1 --model meta-llama/Llama-3-8B
```
This script:
- Loads the processed dataset for the specified fold
- Benchmarks the model using MT-Bench before training
- Initializes Llama-3 model with 8-bit quantization
- Applies LoRA for efficient fine-tuning
- Trains the model on the HelpSteer2 dataset
- Evaluates on the test set
- Benchmarks the model after training
- Saves the trained teacher model

### 3. Knowledge Distillation to Llama-2

```bash
python train_student_model.py --fold 1 --teacher_path llama3-helpsteer-teacher --student_model meta-llama/Llama-2-7b-hf
```
This script:
- Loads the trained Llama-3 teacher model
- Initializes a Llama-2 model as the student
- Benchmarks the student model before training
- Performs knowledge distillation from Llama-3 to Llama-2
- Evaluates the student model on the test set
- Compares model sizes and performance
- Benchmarks the student model after training
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
- Base Model: Llama-3-8B
- Quantization: 8-bit
- Fine-tuning: LoRA
- Task: Helpfulness classification on HelpSteer2 dataset

### Student Model
- Base Model: Llama-2-7b
- Training: Knowledge Distillation
- Quantization: ONNX Runtime
- Size: Smaller than teacher model while maintaining good performance

## K-Fold Cross-Validation

The project uses 10-fold cross-validation:
- Original training data is split into 10 folds
- For each fold, 90% of the data is used for training and 10% for validation
- The original validation set is used as the holdout test set
- This approach provides more robust model evaluation

## Benchmarking

The project uses MT-Bench to benchmark models:
- Before training to establish a baseline
- After training to measure improvement
- Results are saved for comparison

## Performance Metrics

The project tracks several metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- MT-Bench scores

## Additional Notes

1. **Memory Requirements**:
   - Teacher training: ~24GB GPU RAM (with 8-bit quantization)
   - Distillation: ~16GB GPU RAM
   - Quantized student inference: ~4GB RAM

2. **Training Time**:
   - Teacher: ~6-8 hours on a single A100
   - Distillation: ~4-6 hours on a single A100
   - Quantization: ~15 minutes

3. **Model Sizes**:
   - Teacher (Llama-3-8B): ~8GB
   - Student (Llama-2-7b): ~7GB
   - Quantized Student: ~1.8GB

## License

This project is licensed under the MIT License - see the LICENSE file for details.