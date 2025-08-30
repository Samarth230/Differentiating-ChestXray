# 🏥 Medical Image Classification System

**Normal, Pneumonia, and Tuberculosis Detection using Deep Learning**

A comprehensive deep learning system for automated chest X-ray classification, designed to assist medical professionals in faster and more accurate diagnoses while maintaining high ethical standards and interpretability.

## 🌟 Features

### **Core Functionality**
- **Multi-Class Classification**: Detects Normal, Pneumonia, and Tuberculosis from chest X-rays
- **Advanced Architecture**: EfficientNet-B0 backbone with attention mechanisms
- **Transfer Learning**: Pre-trained on ImageNet for robust feature extraction
- **Uncertainty Estimation**: Monte Carlo dropout for prediction confidence
- **Attention Maps**: Visual interpretability for medical decision support

### **Technical Excellence**
- **Data Augmentation**: Comprehensive image transformations for robust training
- **Regularization**: Dropout, BatchNorm, Weight Decay, and Gradient Clipping
- **Optimization**: AdamW optimizer with Cosine Annealing scheduler
- **Early Stopping**: Prevents overfitting with configurable patience
- **Checkpointing**: Automatic model saving and restoration

### **Ethical AI & Fairness**
- **Bias Detection**: Comprehensive analysis across demographic attributes
- **Fairness Metrics**: Statistical Parity, Equalized Odds, Equal Opportunity
- **Adversarial Debiasing**: Built-in bias mitigation strategies
- **Transparency**: Detailed reporting and interpretability tools
- **Clinical Validation**: Designed for real-world medical applications

### **Evaluation & Monitoring**
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Visualization**: Confusion matrices, ROC curves, precision-recall curves
- **TensorBoard Integration**: Real-time training monitoring
- **Performance Analysis**: Per-class breakdown and confidence analysis
- **Ethical Reports**: Automated bias detection and fairness analysis

## 📋 Requirements

### **System Requirements**
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space for datasets and models
- **GPU**: CUDA-compatible GPU recommended (not required)

### **Python Dependencies**
- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ with torchvision
- **Deep Learning**: timm, albumentations
- **Data Science**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Medical Imaging**: Pillow, opencv-python
- **Interpretability**: SHAP, LIME
- **Monitoring**: tensorboard, tqdm

## 🚀 Installation

### **1. Clone the Repository**
```bash
git clone <your-repository-url>
cd Envision_AIML
```

### **2. Create Virtual Environment (Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Verify Installation**
```bash
python main.py --info
```

## 📊 Dataset Preparation

### **Required Structure**
```
data/
├── normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── tuberculosis/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### **Image Requirements**
- **Format**: JPG, JPEG, PNG, BMP, TIFF
- **Size**: Any size (automatically resized to 224x224)
- **Channels**: RGB (3 channels)
- **Quality**: Medical-grade chest X-rays
- **Minimum**: 100 images per class (recommended: 1000+)

### **Data Sources**
- **Normal**: Healthy chest X-rays
- **Pneumonia**: Bacterial/viral pneumonia cases
- **Tuberculosis**: Confirmed TB cases
- **Ethical Note**: Ensure proper consent and anonymization

## 🎯 Usage

### **Quick Start**
```bash
# Train the model
python main.py --train

# Evaluate the trained model
python main.py --evaluate

# Make predictions on new images
python main.py --predict --images "path/to/image.jpg"

# Process entire folders
python main.py --predict --images "test_folder/"

# Show model information
python main.py --info

# Run full pipeline (train + evaluate)
python main.py --full-pipeline
```

### **Advanced Usage**
```bash
# Custom configuration
python main.py --train --config-file custom_config.json

# Custom model path
python main.py --evaluate --model-path "models/my_model.pth"

# Multiple image prediction
python main.py --predict --images "img1.jpg" "img2.jpg" "folder/"

# Custom data directory
python main.py --train --data-dir "my_dataset/"
```

### **Configuration Options**
```bash
# Available configurations
python main.py --train --config complete      # Full feature set
python main.py --train --config basic         # Essential features only
python main.py --train --config custom        # Custom config file
```

## 🔧 Configuration

### **Model Configuration**
```json
{
  "model": {
    "model_type": "efficientnet_b0",
    "num_classes": 3,
    "pretrained": true,
    "dropout_rate": 0.3,
    "use_ensemble": false,
    "use_fairness": true,
    "sensitive_attributes": ["age", "gender"]
  }
}
```

### **Training Configuration**
```json
{
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "early_stopping_patience": 15
  }
}
```

### **Data Configuration**
```json
{
  "data": {
    "img_size": 224,
    "test_size": 0.15,
    "val_size": 0.15,
    "augmentation": {
      "horizontal_flip_p": 0.5,
      "rotation_p": 0.3,
      "blur_p": 0.3
    }
  }
}
```

## 📈 Training Process

### **Automatic Data Splitting**
- **Training**: 70% of data
- **Validation**: 15% of data  
- **Testing**: 15% of data
- **Stratified**: Maintains class balance across splits

### **Training Features**
- **Progress Monitoring**: Real-time loss and accuracy tracking
- **Validation**: Automatic validation after each epoch
- **Checkpointing**: Saves best model based on validation performance
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

### **Monitoring**
```bash
# View training progress
tensorboard --logdir outputs/logs

# Check saved models
ls outputs/
# - best_model.pth (best validation performance)
# - latest_model.pth (most recent checkpoint)
# - training_history.json (training metrics)
```

## 📊 Evaluation & Results

### **Comprehensive Metrics**
- **Overall Performance**: Accuracy, Precision, Recall, F1-score
- **Per-Class Analysis**: Individual class performance breakdown
- **ROC Analysis**: AUC-ROC curves for each class
- **Precision-Recall**: Detailed classification performance
- **Confusion Matrix**: Visual error analysis

### **Output Files**
```
evaluation_results/
├── confusion_matrix.png
├── roc_curves.png
├── precision_recall_curves.png
├── classification_report.json
├── performance_summary.csv
└── ethical_analysis_report.txt
```

### **Ethical Analysis**
- **Bias Detection**: Demographic and geographic bias analysis
- **Fairness Metrics**: Statistical parity and equalized odds
- **Recommendations**: Bias mitigation strategies
- **Transparency**: Detailed reporting for clinical review

## 🔮 Prediction & Inference

### **Single Image Prediction**
```bash
python main.py --predict --images "chest_xray.jpg"
```

### **Batch Processing**
```bash
# Process multiple images
python main.py --predict --images "img1.jpg" "img2.jpg" "img3.jpg"

# Process entire folders
python main.py --predict --images "patient_images/"
```

### **Output Format**
```json
{
  "predicted_class": "Pneumonia",
  "confidence": 0.892,
  "probabilities": {
    "Normal": 0.023,
    "Pneumonia": 0.892,
    "Tuberculosis": 0.085
  },
  "uncertainty": 0.045
}
```

### **Visualization**
- **Prediction Plots**: Class probabilities and confidence scores
- **Attention Maps**: Model focus areas (when available)
- **Batch Reports**: CSV summaries for multiple predictions

## 🏗️ Project Structure

```
Envision_AIML/
├── main.py                 # Main entry point
├── config.py              # Configuration management
├── data_preprocessing.py  # Data loading and augmentation
├── model.py              # Neural network architectures
├── train.py              # Training pipeline
├── evaluate.py           # Model evaluation
├── predict.py            # Prediction and inference
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── ETHICAL_CONSIDERATIONS.md  # Ethical framework
├── SETUP_AND_USAGE.md   # Detailed setup guide
├── data/                # Dataset directory
├── outputs/             # Model checkpoints and logs
├── evaluation_results/  # Evaluation outputs
└── prediction_results/  # Prediction outputs
```

## 🧪 Testing & Validation

### **Unit Tests**
```bash
# Test individual components
python -m pytest tests/

# Test specific module
python -m pytest tests/test_model.py
```

### **Integration Tests**
```bash
# Test full pipeline
python main.py --full-pipeline

# Test prediction pipeline
python main.py --predict --images "test_images/"
```

### **Performance Validation**
- **Cross-Validation**: K-fold validation for robust performance estimation
- **Out-of-Distribution**: Test on different demographic groups
- **Clinical Validation**: Designed for real-world medical applications

## 🚨 Troubleshooting

### **Common Issues**

#### **"No module named timm"**
```bash
pip install timm
```

#### **CUDA Out of Memory**
- Reduce batch size in config
- Use CPU-only mode
- Process smaller image batches

#### **Data Loading Errors**
- Verify data directory structure
- Check image file formats
- Ensure sufficient images per class

#### **Model Loading Issues**
- Verify checkpoint file exists
- Check PyTorch version compatibility
- Ensure model architecture matches

### **Performance Optimization**
```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Optimize data loading
# Increase num_workers in config (if RAM allows)

# Memory optimization
# Reduce batch_size or image_size
```

## 🔒 Ethical Considerations

### **Medical Disclaimer**
⚠️ **This system is for research and educational purposes only. Clinical decisions should always be made by qualified medical professionals.**

### **Privacy & Security**
- **Data Anonymization**: No personally identifiable information
- **Secure Processing**: Local processing, no data transmission
- **Consent Requirements**: Proper patient consent for data usage
- **HIPAA Compliance**: Designed with medical privacy standards in mind

### **Bias & Fairness**
- **Demographic Bias**: Automatic detection and reporting
- **Geographic Bias**: Multi-center validation recommendations
- **Temporal Bias**: Dataset age and relevance monitoring
- **Mitigation Strategies**: Built-in bias reduction techniques

### **Transparency**
- **Interpretability**: Attention maps and uncertainty estimation
- **Documentation**: Comprehensive technical and ethical documentation
- **Validation**: Multiple evaluation metrics and fairness analysis
- **Reproducibility**: Version control and configuration management

## 📚 References & Citations

### **Technical Papers**
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- Attention Mechanisms in Medical Image Analysis
- Fairness in Machine Learning for Healthcare

### **Datasets**
- Chest X-ray datasets (ensure proper attribution)
- Medical imaging benchmarks
- Clinical validation studies

### **Libraries & Tools**
- PyTorch: Deep learning framework
- timm: Image models and utilities
- albumentations: Image augmentation
- scikit-learn: Machine learning utilities

## 🤝 Contributing

### **Development Guidelines**
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** changes with tests
4. **Ensure** ethical compliance
5. **Submit** a pull request

### **Code Standards**
- **Python**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests
- **Type Hints**: Full type annotation

### **Ethical Review**
- **Bias Analysis**: All changes must include bias assessment
- **Clinical Validation**: Medical applications require clinical review
- **Transparency**: Clear documentation of all modifications

## 📄 License

### **Academic Use**
- **Research**: Free for academic and research purposes
- **Attribution**: Proper citation required
- **Modification**: Allowed with ethical review

### **Commercial Use**
- **Clinical Deployment**: Requires clinical validation and regulatory approval
- **Commercial Licensing**: Contact for commercial licensing terms
- **Compliance**: Must meet medical device regulations

### **Open Source**
- **Code**: Available under open source license
- **Models**: Pre-trained models for research use
- **Documentation**: Comprehensive guides and tutorials

## 📞 Support & Contact

### **Technical Support**
- **Issues**: GitHub issue tracker
- **Documentation**: Comprehensive guides and tutorials
- **Community**: Academic and research community support

### **Clinical Support**
- **Medical Validation**: Clinical validation guidelines
- **Regulatory Compliance**: Medical device regulations
- **Clinical Integration**: Healthcare system integration

### **Research Collaboration**
- **Academic Partnerships**: University and research institution collaboration
- **Clinical Studies**: Clinical validation and research studies
- **Data Sharing**: Ethical data sharing frameworks

## 🎓 Academic Use

### **Research Applications**
- **Medical AI Research**: Deep learning in medical imaging
- **Clinical Studies**: Validation and comparison studies
- **Educational Purposes**: Medical AI education and training

### **Publication Guidelines**
- **Citation**: Proper attribution to this work
- **Ethical Review**: Include ethical considerations in publications
- **Clinical Validation**: Emphasize research vs. clinical use

### **Collaboration Opportunities**
- **Multi-Center Studies**: Collaborative clinical validation
- **Benchmark Development**: Medical AI benchmarking
- **Open Science**: Reproducible research practices

---

**Built with ❤️ for the medical AI community**

*This project represents a comprehensive approach to medical image classification, combining state-of-the-art deep learning techniques with robust ethical considerations and clinical validation frameworks.*
