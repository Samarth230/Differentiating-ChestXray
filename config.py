"""
Configuration file for Medical Image Classification Project
"""

DATA_CONFIG = {
    'img_size': 224,
    'test_size': 0.15,
    'val_size': 0.15,
    'num_workers': 4,
    'pin_memory': True
}

MODEL_CONFIG = {
    'model_type': 'efficientnet_b0',
    'num_classes': 3,
    'pretrained': True,
    'dropout_rate': 0.3,
    'use_ensemble': False,
    'use_fairness': True,
    'sensitive_attributes': ['age', 'gender'],
    'attention_reduction': 16
}

TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'gradient_clipping': True,
    'max_grad_norm': 1.0,
    'use_class_weights': True,
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 1e-4,
    'output_dir': 'outputs',
    'save_frequency': 5,
    'log_frequency': 10
}

OPTIMIZER_CONFIG = {
    'type': 'adamw',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'nesterov': True
}

SCHEDULER_CONFIG = {
    'type': 'cosine',
    'epochs': 50,
    'min_lr': 1e-6,
    'step_size': 10,
    'gamma': 0.1,
    'factor': 0.5,
    'patience': 5
}

AUGMENTATION_CONFIG = {
    'train': {
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.1,
        'rotation_p': 0.3,
        'rotation_limit': 15,
        'shift_scale_rotate_p': 0.5,
        'blur_p': 0.3,
        'distortion_p': 0.3,
        'color_p': 0.3
    },
    'validation': {
        'horizontal_flip_p': 0.0,
        'vertical_flip_p': 0.0,
        'rotation_p': 0.0,
        'rotation_limit': 0,
        'shift_scale_rotate_p': 0.0,
        'blur_p': 0.0,
        'distortion_p': 0.0,
        'color_p': 0.0
    }
}

EVALUATION_CONFIG = {
    'batch_size': 32,
    'uncertainty_samples': 10,
    'confidence_threshold': 0.8,
    'save_predictions': True,
    'create_visualizations': True,
    'output_dir': 'evaluation_results'
}

ETHICAL_CONFIG = {
    'bias_detection': True,
    'fairness_metrics': True,
    'sensitive_attributes': ['age', 'gender', 'ethnicity'],
    'bias_thresholds': {
        'high_bias': 0.05,
        'moderate_bias': 0.02,
        'low_bias': 0.01
    },
    'adversarial_debiasing': True,
    'fairness_constraints': ['demographic_parity', 'equalized_odds']
}

INTERPRETABILITY_CONFIG = {
    'attention_maps': True,
    'grad_cam': True,
    'shap_values': True,
    'lime_explanations': True,
    'uncertainty_estimation': True,
    'confidence_calibration': True
}

LOGGING_CONFIG = {
    'tensorboard': True,
    'wandb': False,
    'log_metrics': True,
    'log_gradients': False,
    'log_hyperparameters': True,
    'log_model_graph': True
}

HARDWARE_CONFIG = {
    'device': 'auto',
    'mixed_precision': True,
    'num_gpus': 1,
    'distributed_training': False
}

COMPLETE_CONFIG = {
    'data': DATA_CONFIG,
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'optimizer': OPTIMIZER_CONFIG,
    'scheduler': SCHEDULER_CONFIG,
    'augmentation': AUGMENTATION_CONFIG,
    'evaluation': EVALUATION_CONFIG,
    'ethical': ETHICAL_CONFIG,
    'interpretability': INTERPRETABILITY_CONFIG,
    'logging': LOGGING_CONFIG,
    'hardware': HARDWARE_CONFIG
}

DATASET_CONFIGS = {
    'nih_chest_xray': {
        'name': 'NIH Chest X-ray Dataset',
        'url': 'https://www.kaggle.com/datasets/nih-chest-xrays/data',
        'classes': ['Normal', 'Pneumonia', 'Tuberculosis'],
        'expected_samples': 112120,
        'image_format': 'png',
        'metadata_columns': ['age', 'gender', 'ethnicity', 'hospital']
    },
    'chexpert': {
        'name': 'CheXpert Dataset',
        'url': 'https://stanfordmlgroup.github.io/competitions/chexpert/',
        'classes': ['Normal', 'Pneumonia', 'Tuberculosis'],
        'expected_samples': 224316,
        'image_format': 'jpg',
        'metadata_columns': ['age', 'gender', 'ethnicity', 'hospital']
    },
    'mimic_cxr': {
        'name': 'MIMIC-CXR Dataset',
        'url': 'https://physionet.org/content/mimic-cxr/2.0.0/',
        'classes': ['Normal', 'Pneumonia', 'Tuberculosis'],
        'expected_samples': 377110,
        'image_format': 'jpg',
        'metadata_columns': ['age', 'gender', 'ethnicity', 'hospital']
    }
}

MODEL_ARCHITECTURES = {
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'params': '5.3M',
        'input_size': 224,
        'feature_dim': 1280,
        'pretrained': True
    },
    'efficientnet_b1': {
        'name': 'EfficientNet-B1',
        'params': '7.8M',
        'input_size': 240,
        'feature_dim': 1280,
        'pretrained': True
    },
    'efficientnet_b2': {
        'name': 'EfficientNet-B2',
        'params': '9.2M',
        'input_size': 260,
        'feature_dim': 1408,
        'pretrained': True
    },
    'resnet50': {
        'name': 'ResNet-50',
        'params': '25.6M',
        'input_size': 224,
        'feature_dim': 2048,
        'pretrained': True
    }
}

PERFORMANCE_BENCHMARKS = {
    'accuracy_threshold': 0.85,
    'f1_threshold': 0.80,
    'auc_threshold': 0.90,
    'bias_threshold': 0.02,
    'uncertainty_threshold': 0.15
}

VALIDATION_METRICS = {
    'primary': 'macro_f1',
    'secondary': ['accuracy', 'macro_auc', 'bias_score'],
    'thresholds': {
        'macro_f1': 0.80,
        'accuracy': 0.85,
        'macro_auc': 0.90,
        'bias_score': 0.02
    }
}

def get_config(config_name: str = 'complete'):
    """Get configuration by name"""
    configs = {
        'complete': COMPLETE_CONFIG,
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'optimizer': OPTIMIZER_CONFIG,
        'scheduler': SCHEDULER_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'ethical': ETHICAL_CONFIG,
        'interpretability': INTERPRETABILITY_CONFIG,
        'logging': LOGGING_CONFIG,
        'hardware': HARDWARE_CONFIG
    }
    
    return configs.get(config_name, COMPLETE_CONFIG)

def validate_config(config: dict) -> bool:
    """Validate configuration parameters"""
    errors = []
    
    # Check required fields
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate data configuration
    if 'data' in config:
        data = config['data']
        if data.get('img_size', 0) <= 0:
            errors.append("Image size must be positive")
        if not (0 < data.get('test_size', 0) < 1):
            errors.append("Test size must be between 0 and 1")
        if not (0 < data.get('val_size', 0) < 1):
            errors.append("Validation size must be between 0 and 1")
    
    # Validate model configuration
    if 'model' in config:
        model = config['model']
        if model.get('num_classes', 0) <= 0:
            errors.append("Number of classes must be positive")
        if model.get('dropout_rate', 0) < 0 or model.get('dropout_rate', 0) > 1:
            errors.append("Dropout rate must be between 0 and 1")
    
    # Validate training configuration
    if 'training' in config:
        training = config['training']
        if training.get('epochs', 0) <= 0:
            errors.append("Number of epochs must be positive")
        if training.get('batch_size', 0) <= 0:
            errors.append("Batch size must be positive")
        if training.get('learning_rate', 0) <= 0:
            errors.append("Learning rate must be positive")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def print_config_summary(config: dict):
    """Print a summary of the configuration"""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    # Data
    if 'data' in config:
        data = config['data']
        print(f"Data Configuration:")
        print(f"  Image size: {data.get('img_size', 'N/A')}")
        print(f"  Test split: {data.get('test_size', 'N/A')}")
        print(f"  Validation split: {data.get('val_size', 'N/A')}")
        print()
    
    # Model
    if 'model' in config:
        model = config['model']
        print(f"Model Configuration:")
        print(f"  Architecture: {model.get('model_type', 'N/A')}")
        print(f"  Classes: {model.get('num_classes', 'N/A')}")
        print(f"  Pretrained: {model.get('pretrained', 'N/A')}")
        print(f"  Dropout rate: {model.get('dropout_rate', 'N/A')}")
        print(f"  Fairness-aware: {model.get('use_fairness', 'N/A')}")
        print()
    
    # Training
    if 'training' in config:
        training = config['training']
        print(f"Training Configuration:")
        print(f"  Epochs: {training.get('epochs', 'N/A')}")
        print(f"  Batch size: {training.get('batch_size', 'N/A')}")
        print(f"  Learning rate: {training.get('learning_rate', 'N/A')}")
        print(f"  Output directory: {training.get('output_dir', 'N/A')}")
        print()
    
    # Ethical considerations
    if 'ethical' in config:
        ethical = config['ethical']
        print(f"Ethical AI Configuration:")
        print(f"  Bias detection: {ethical.get('bias_detection', 'N/A')}")
        print(f"  Fairness metrics: {ethical.get('fairness_metrics', 'N/A')}")
        print(f"  Sensitive attributes: {ethical.get('sensitive_attributes', 'N/A')}")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    # Example usage
    config = get_config('complete')
    
    if validate_config(config):
        print("Configuration is valid!")
        print_config_summary(config)
    else:
        print("Configuration has errors!")
