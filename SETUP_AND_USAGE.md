# Setup and Usage Guide

## Medical Image Classification: Normal, Pneumonia, and Tuberculosis Detection

This guide will walk you through setting up and using our comprehensive medical image classification system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

## System Requirements

### Hardware Requirements

- **CPU**: Intel i5/AMD Ryzen 5 or better (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better recommended)
- **Storage**: 50GB+ free space for datasets and models
- **Network**: Stable internet connection for downloading datasets

### Software Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.0+ (for GPU acceleration)
- **Git**: For version control

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Envision_AIML
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n medical_ai python=3.9
conda activate medical_ai

# Or using venv
python -m venv medical_ai
source medical_ai/bin/activate  # On Windows: medical_ai\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
```

## Data Preparation

### Dataset Structure

Organize your chest X-ray images in the following structure:

```
data/
├── normal/
│   ├── normal_001.jpg
│   ├── normal_002.jpg
│   └── ...
├── pneumonia/
│   ├── pneumonia_001.jpg
│   ├── pneumonia_002.jpg
│   └── ...
└── tuberculosis/
    ├── tuberculosis_001.jpg
    ├── tuberculosis_002.jpg
    └── ...
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)

### Recommended Datasets

1. **NIH Chest X-ray Dataset**
   - Download: [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
   - Size: ~112GB
   - Images: ~112,000

2. **CheXpert Dataset**
   - Download: [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)
   - Size: ~500GB
   - Images: ~224,000

3. **MIMIC-CXR Dataset**
   - Download: [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)
   - Size: ~1TB
   - Images: ~377,000

### Data Preprocessing

The system automatically handles:
- Image resizing to 224x224 pixels
- Normalization using ImageNet statistics
- Data augmentation (training only)
- Train/validation/test splitting (70/15/15)

## Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python main.py --train

# Train with custom data directory
python main.py --train --data-dir /path/to/your/data
```

### 2. Evaluate Trained Model

```bash
# Evaluate with default model path
python main.py --evaluate

# Evaluate with custom model path
python main.py --evaluate --model-path /path/to/model.pth
```

### 3. Make Predictions

```bash
# Predict on single image
python main.py --predict --images image1.jpg

# Predict on multiple images
python main.py --predict --images image1.jpg image2.jpg image3.jpg
```

### 4. Full Pipeline

```bash
# Train and evaluate in one command
python main.py --full-pipeline
```

## Detailed Usage

### Training Configuration

#### Basic Training

```python
from config import get_config
from train import MedicalImageTrainer

# Get configuration
config = get_config('complete')

# Initialize trainer
trainer = MedicalImageTrainer(config)

# Prepare data
trainer.prepare_data("data/")

# Start training
trainer.train()
```

#### Custom Training Parameters

```python
# Modify configuration
config['training']['epochs'] = 100
config['training']['batch_size'] = 64
config['training']['learning_rate'] = 5e-5

# Use custom optimizer
config['training']['optimizer'] = {
    'type': 'sgd',
    'lr': 1e-3,
    'momentum': 0.9,
    'weight_decay': 1e-4
}
```

### Model Architecture Selection

```python
# EfficientNet-B0 (default, fastest)
config['model']['model_type'] = 'efficientnet_b0'

# EfficientNet-B1 (balanced)
config['model']['model_type'] = 'efficientnet_b1'

# ResNet-50 (most robust)
config['model']['model_type'] = 'resnet50'

# Ensemble (best performance)
config['model']['use_ensemble'] = True
```

### Data Augmentation

```python
# Custom augmentation
config['augmentation']['train']['horizontal_flip_p'] = 0.7
config['augmentation']['train']['rotation_limit'] = 30
config['augmentation']['train']['blur_p'] = 0.5
```

### Evaluation and Analysis

#### Comprehensive Evaluation

```python
from evaluate import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(config, "outputs/best_model.pth")

# Prepare test data
evaluator.prepare_test_data("data/")

# Evaluate model
evaluator.evaluate_model()

# Generate reports
evaluator.create_visualizations("evaluation_results/")
evaluator.generate_detailed_report("evaluation_results/")
evaluator.analyze_predictions("evaluation_results/")
```

#### Fairness Analysis

```python
# Analyze bias across demographic groups
fairness_results = evaluator.analyze_fairness()

# View bias report
print(fairness_results['bias_report'])

# View ethical analysis
print(fairness_results['ethical_report'])
```

### Prediction and Inference

#### Single Image Prediction

```python
from predict import MedicalImagePredictor

# Initialize predictor
predictor = MedicalImagePredictor(config, "outputs/best_model.pth")

# Make prediction
result = predictor.predict_single_image("image.jpg", return_attention=True)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probabilities: {result['class_probabilities']}")
```

#### Uncertainty Estimation

```python
# Predict with uncertainty
uncertainty_result = predictor.predict_with_uncertainty("image.jpg", num_samples=20)

print(f"Prediction uncertainty: {uncertainty_result['prediction_uncertainty']:.4f}")
print(f"Model uncertainty: {uncertainty_result['model_uncertainty']:.4f}")
print(f"Total uncertainty: {uncertainty_result['total_uncertainty']:.4f}")
```

#### Batch Prediction

```python
# Predict on multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = predictor.predict_batch(image_paths)

# Create report
predictor.create_prediction_report(results, "batch_report.csv")
```

## Configuration

### Configuration Files

The system uses a hierarchical configuration system:

1. **Built-in Configurations**: Pre-defined configurations in `config.py`
2. **Custom Configuration Files**: JSON files with your settings
3. **Command-line Overrides**: Modify specific parameters at runtime

### Built-in Configurations

```bash
# Complete configuration (recommended)
python main.py --config complete

# Data-only configuration
python main.py --config data

# Model-only configuration
python main.py --config model

# Training-only configuration
python main.py --config training
```

### Custom Configuration File

Create `custom_config.json`:

```json
{
    "data": {
        "img_size": 256,
        "test_size": 0.20,
        "val_size": 0.20
    },
    "model": {
        "model_type": "efficientnet_b1",
        "dropout_rate": 0.5
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 5e-5
    }
}
```

Use custom configuration:

```bash
python main.py --train --config-file custom_config.json
```

### Environment Variables

Set environment variables for customization:

```bash
export MEDICAL_AI_DATA_DIR="/path/to/data"
export MEDICAL_AI_OUTPUT_DIR="/path/to/outputs"
export MEDICAL_AI_MODEL_PATH="/path/to/model"
export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU
```

## Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs

# View in browser: http://localhost:6006
```

### Log Files

The system generates comprehensive logs:

- **Training Logs**: `outputs/logs/`
- **Evaluation Results**: `evaluation_results/`
- **Prediction Results**: `prediction_results/`
- **Model Checkpoints**: `outputs/best_model.pth`

### Performance Metrics

Key metrics tracked:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Bias Score**: Fairness across demographic groups
- **Uncertainty**: Model confidence in predictions

## Advanced Features

### Ensemble Models

```python
# Enable ensemble training
config['model']['use_ensemble'] = True

# Custom ensemble weights
config['model']['ensemble_weights'] = [0.4, 0.3, 0.3]
```

### Fairness-Aware Training

```python
# Enable fairness constraints
config['model']['use_fairness'] = True
config['model']['sensitive_attributes'] = ['age', 'gender', 'ethnicity']

# Set fairness thresholds
config['ethical']['bias_thresholds']['high_bias'] = 0.03
```

### Interpretability Features

```python
# Enable all interpretability features
config['interpretability']['attention_maps'] = True
config['interpretability']['grad_cam'] = True
config['interpretability']['shap_values'] = True
config['interpretability']['lime_explanations'] = True
```

### Mixed Precision Training

```python
# Enable mixed precision for faster training
config['hardware']['mixed_precision'] = True
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
config['training']['batch_size'] = 16

# Use gradient accumulation
config['training']['gradient_accumulation_steps'] = 2

# Enable mixed precision
config['hardware']['mixed_precision'] = True
```

#### 2. Slow Training

```bash
# Increase batch size (if memory allows)
config['training']['batch_size'] = 64

# Use multiple GPUs
config['hardware']['num_gpus'] = 2

# Enable distributed training
config['hardware']['distributed_training'] = True
```

#### 3. Poor Performance

```bash
# Check data quality
python main.py --info

# Verify data distribution
python data_preprocessing.py

# Try different model architecture
config['model']['model_type'] = 'efficientnet_b1'
```

#### 4. Import Errors

```bash
# Verify installation
pip list | grep torch
pip list | grep timm

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Optimization

#### GPU Optimization

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES="0"

# Enable TensorRT (if available)
config['hardware']['use_tensorrt'] = True

# Optimize memory usage
config['hardware']['memory_efficient'] = True
```

#### Data Loading Optimization

```bash
# Increase number of workers
config['data']['num_workers'] = 8

# Enable pin memory
config['data']['pin_memory'] = True

# Use prefetch factor
config['data']['prefetch_factor'] = 2
```

## Best Practices

### 1. Data Quality

- Ensure balanced class distribution
- Validate image quality and annotations
- Remove corrupted or low-quality images
- Use consistent image formats and sizes

### 2. Training Strategy

- Start with pre-trained models
- Use learning rate scheduling
- Implement early stopping
- Monitor for overfitting

### 3. Evaluation

- Use multiple metrics (not just accuracy)
- Perform cross-validation
- Test on external datasets
- Analyze failure cases

### 4. Deployment

- Validate model performance thoroughly
- Implement monitoring and alerting
- Plan for model updates
- Maintain human oversight

## Support and Resources

### Documentation

- **API Reference**: See docstrings in source code
- **Examples**: Check `examples/` directory
- **Tutorials**: Follow step-by-step guides

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share solutions
- **Contributions**: Submit improvements and fixes

### Citation

If you use this system in your research, please cite:

```bibtex
@software{medical_image_classification,
  title={Medical Image Classification: Normal, Pneumonia, and Tuberculosis Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Envision_AIML}
}
```

---

**Note**: This system is designed for research and educational purposes. For clinical use, additional validation, regulatory compliance, and clinical testing are required. Always consult with medical professionals before deploying AI systems in clinical settings.
